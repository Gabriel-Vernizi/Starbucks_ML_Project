from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# ------------------------- #

class HistoryEnricher(BaseEstimator, TransformerMixin):
    def __init__(self, full_log_df, threshold_percentile=0.75):
        """
        full_log_df: DataFrame com histórico de transações.
        threshold_percentile: Define o corte para 'High Spender' (0.75 = Top 25%).
        """
        self.full_log_df = full_log_df.copy()
        self.threshold_percentile = threshold_percentile
        
        self.history_features = None 
        self.spending_threshold = 0 # Valor que será aprendido no fit

        self.X_enriched = None

    def fit(self, X, y=None):
        # 1. PREPARAÇÃO DA TABELA DE BUSCA (LOOKUP TABLE)
        df_trans = self.full_log_df[self.full_log_df['event'] == 'transaction'].copy()
        df_trans = df_trans.sort_values(by=['costumer_id', 'time'])
        
        # Features Temporais
        df_trans['prev_trans_time'] = df_trans.groupby('costumer_id')['time'].shift(1).fillna(0)
        df_trans['hours_since_last_buy'] = df_trans['time'] - df_trans['prev_trans_time']
        
        df_trans['avg_inter_purchase_hours'] = df_trans.groupby('costumer_id')['hours_since_last_buy'] \
                                                       .expanding().mean() \
                                                       .reset_index(level=0, drop=True)
        
        # Feature: Total Gasto e Contagem
        df_trans['total_spent_so_far'] = df_trans.groupby('costumer_id')['amount'].cumsum()
        df_trans['trans_count_so_far'] = df_trans.groupby('costumer_id').cumcount() + 1
        
        # Feature: Ticket Médio
        df_trans['avg_ticket_so_far'] = df_trans['total_spent_so_far'] / df_trans['trans_count_so_far']

        # Salva a tabela para uso no transform
        self.history_features = df_trans[['costumer_id', 'time', 
                                          'avg_inter_purchase_hours', 
                                          'total_spent_so_far',
                                          'trans_count_so_far',
                                          'avg_ticket_so_far']].sort_values('time')
        
        # Aprendizado do Threshold
        X_sorted = X.sort_values('time')
        
        # Simula o cenário real no treino
        merged_train = pd.merge_asof(
            X_sorted, 
            self.history_features, 
            on='time', 
            by='costumer_id', 
            direction='backward',
            allow_exact_matches=False
        )
        
        # Aprendemos o corte com base na distribuição do TREINO
        valid_spent = merged_train['total_spent_so_far'].dropna()

        if not valid_spent.empty:
            self.spending_threshold = valid_spent.quantile(self.threshold_percentile)
        
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # Salva índice e ordena
        X_copy['_orig_index'] = X_copy.index
        X_copy = X_copy.sort_values('time')
        
        # Merge Inteligente (Point-in-Time Join)
        X_enriched = pd.merge_asof(
            X_copy,
            self.history_features, #type: ignore
            on='time',
            by='costumer_id',
            direction='backward',
            allow_exact_matches=False
        )
        
        # Tratamento de Nulos (Novos Clientes)
        fill_values = {
            'avg_inter_purchase_hours': -1,
            'total_spent_so_far': 0,
            'trans_count_so_far': 0,
            'avg_ticket_so_far': 0 # Quem nunca comprou tem ticket 0
        }
        X_enriched = X_enriched.fillna(fill_values)
        
        # Feature: High Spender
        X_enriched['is_high_spender'] = (X_enriched['total_spent_so_far'] > self.spending_threshold).astype(int)
        
        X_enriched = X_enriched.sort_values('_orig_index').drop(columns=['_orig_index'])
        
        self.X_enriched = X_enriched

        return X_enriched

# ------------------------- #

class CustomImputer_for_Gender_Age_Income(BaseEstimator, TransformerMixin):
    def __init__(self, spending_col='total_spent_so_far', spending_threshold=20):
        """
        spending_col: Nome da coluna que indica gasto (se existir no X) para ajudar a decidir o gênero.
        spending_threshold: Valor de corte. Se gasto > x e gênero nulo, assumimos um gênero que gasta mais (ex: F).
        """
        self.fill_values = dict()
        self.spending_col = spending_col
        self.spending_threshold = spending_threshold
        
        # Placeholders para as medianas aprendidas
        self.age_medians = {}
        self.income_medians = {}
        self.gender_mode = 'M' # Fallback padrão

    def fit(self, X, y=None):
        df = X.copy()
        
        if 'age' in df.columns:
            df['age'] = df['age'].replace(118, np.nan)
        
        if 'gender' in df.columns:
            self.gender_mode = df['gender'].mode()[0] # Maior Frequência
            
            # 4. Aprender Medianas de Age e Income por gênero
            self.age_medians = df.groupby('gender')['age'].median().to_dict()
            self.income_medians = df.groupby('gender')['income'].median().to_dict()
            
            # Valores globais de fallback (caso apareça um gênero novo no teste)
            self.global_age_median = df['age'].median()
            self.global_income_median = df['income'].median()
            
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        if 'age' in X_copy.columns:
            X_copy['age'] = X_copy['age'].replace(118, np.nan)

        # Imputação
        if 'gender' in X_copy.columns:
            
            # Identifica onde o gênero está faltando
            missing_gender = X_copy['gender'].isnull()
            
            # Se tivermos a coluna de gastos no dataset, usamos sua regra personalizada
            if self.spending_col in X_copy.columns:
                # Regra: Se Gênero Null E Gasto > Threshold -> Imputa 'F' (ou 'O')
                high_spender_mask = missing_gender & (X_copy[self.spending_col] > self.spending_threshold)
                low_spender_mask = missing_gender & (X_copy[self.spending_col] <= self.spending_threshold)
                
                X_copy.loc[high_spender_mask, 'gender'] = 'F' 
                X_copy.loc[low_spender_mask, 'gender'] = 'M'
            
            # Preenche qualquer sobra (quem não tinha gasto registrado) com a Moda
            X_copy['gender'] = X_copy['gender'].fillna(self.gender_mode)

        # --- ETAPA 3: IMPUTAÇÃO CONDICIONAL DE AGE E INCOME ---
        # Agora que todo mundo tem gênero, usamos o gênero para definir a renda/idade média
        
        if 'age' in X_copy.columns:
            # Mapeia a mediana correta baseada no gênero da linha
            mapped_age = X_copy['gender'].map(self.age_medians).fillna(self.global_age_median)
            X_copy['age'] = X_copy['age'].fillna(mapped_age)
            
        if 'income' in X_copy.columns:
            # Mapeia a mediana correta baseada no gênero da linha
            mapped_income = X_copy['gender'].map(self.income_medians).fillna(self.global_income_median)
            X_copy['income'] = X_copy['income'].fillna(mapped_income)

        return X_copy
    
# ------------------------- #

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_impute=None, strategy='median',personalized_func=None):
        self.cols_to_impute = cols_to_impute
        self.strategy = strategy
        
        self.personalized_func = personalized_func

        self.fill_values = dict() # Dicionário para guardar as médias/medianas aprendidas

    def fit(self, X, y=None):
        if self.cols_to_impute is None:
            self.cols_to_impute = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in self.cols_to_impute:
            if col in X.columns:
                if self.strategy == 'median':
                    self.fill_values[col] = X[col].median()
                elif self.strategy == 'mean':
                    self.fill_values[col] = X[col].mean()
                elif self.strategy == 'zero':
                    self.fill_values[col] = 0
                    
                elif self.strategy == 'personalized':
                    if self.personalized_func is not None:
                        self.fill_values[col] = self.personalized_func(X[col])
                    else:
                        self.fill_values[col] = 0

        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, value in self.fill_values.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(value)
        return X_copy
    
# ------------------------- #

class ChannelsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, channels_col='channels', remove_cols=None):
        self.channels_col = channels_col
        self.possible_channels = ['web', 'email', 'mobile', 'social']
        
        if remove_cols is None:
            self.remove_cols = []
        elif isinstance(remove_cols, str):
            self.remove_cols = [remove_cols]
        elif isinstance(remove_cols,list):
            self.remove_cols = remove_cols

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        X_copy = X.copy()
        
        # Se a coluna não existir, retorna como está
        if self.channels_col not in X_copy.columns:
            return X_copy
            
        for channel in self.possible_channels:
            if channel in self.remove_cols:
            
            # Cria colunas dummy
                X_copy[f'channel_{channel}'] = X_copy[self.channels_col].apply(
                lambda x: 1 if isinstance(x, (list, str)) and channel in str(x) else 0
            )
        
        return X_copy.drop(columns=[self.channels_col])

# ------------------------- #