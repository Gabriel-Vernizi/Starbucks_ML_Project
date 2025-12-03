from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# ------------------------- #

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_impute=None, strategy='median'):
        self.cols_to_impute = cols_to_impute
        self.strategy = strategy
        self.fill_values = dict() # Dicionário para guardar as médias/medianas aprendidas

    def fit(self, X):
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
        return self

    def transform(self, X):
        # Aplicamos os valores aprendidos (sem olhar para os dados novos)
        X_copy = X.copy()
        for col, value in self.fill_values.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(value)
        return X_copy
    
# ------------------------- #

class ChannelsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, channels_col='channels'):
        self.channels_col = channels_col
        self.possible_channels = ['web', 'email', 'mobile', 'social']

    def fit(self, X, y=None):
        return self # Nada para aprender aqui, as regras são fixas

    def transform(self, X):
        X_copy = X.copy()
        
        # Se a coluna não existir, retorna como está
        if self.channels_col not in X_copy.columns:
            return X_copy
            
        for channel in self.possible_channels:
            # Cria colunas dummy: channel_web, channel_email, etc.
            # Verifica se 'web' está dentro da lista/string daquela linha
            X_copy[f'channel_{channel}'] = X_copy[self.channels_col].apply(
                lambda x: 1 if isinstance(x, (list, str)) and channel in str(x) else 0
            )
            
        # Remove a coluna original 'channels' que o modelo não entende
        return X_copy.drop(columns=[self.channels_col])

# ------------------------- #

class HistoryEnricher(BaseEstimator, TransformerMixin):
    def __init__(self, full_log_df):
        """
        full_log_df: O DataFrame completo (df_merged) contendo o histórico de 'transaction'.
                     Ele serve como a 'memória' para calcular as features passadas.
        """
        self.full_log_df = full_log_df.copy()
        self.history_features = None # Vai guardar a tabela processada

    def fit(self, X, y=None):
        # 1. Filtra apenas as transações do log completo
        df_trans = self.full_log_df[self.full_log_df['event'] == 'transaction'].copy()
        
        # Garante a ordenação temporal (CRUCIAL para cálculos de janela)
        df_trans = df_trans.sort_values(by=['costumer_id', 'time'])

        # 2. Engenharia de Features (Cálculos no histórico)
        
        # Feature A: Tempo desde a última compra (Time Delta)
        df_trans['prev_trans_time'] = df_trans.groupby('costumer_id')['time'].shift(1)
        df_trans['hours_since_last_buy'] = df_trans['time'] - df_trans['prev_trans_time']
        
        # Feature B: Média ACUMULADA de tempo entre compras (Expanding Mean)
        # "Até o momento T, qual é a frequência média de compra desse cliente?"
        df_trans['avg_inter_purchase_hours'] = df_trans.groupby('costumer_id')['hours_since_last_buy'] \
                                                       .expanding().mean() \
                                                       .reset_index(level=0, drop=True)
        
        # Feature C: Total Gasto Acumulado (Customer Lifetime Value dinâmico)
        df_trans['total_spent_so_far'] = df_trans.groupby('costumer_id')['amount'].cumsum()
        
        # Feature D: Contagem de transações acumuladas (Frequência)
        df_trans['trans_count_so_far'] = df_trans.groupby('costumer_id').cumcount() + 1

        # 3. Prepara a tabela de lookup (apenas colunas necessárias + chaves)
        self.history_features = df_trans[['costumer_id', 'time', 
                                          'avg_inter_purchase_hours', 
                                          'total_spent_so_far',
                                          'trans_count_so_far']].sort_values('time')
        
        return self

    def transform(self, X):
        # X aqui é o seu df_model (as ofertas que queremos prever)
        X_copy = X.copy()
        
        # Garante ordenação para o merge_asof
        # Salvamos o índice original para restaurar depois (boa prática em pipelines)
        X_copy['_orig_index'] = X_copy.index
        X_copy = X_copy.sort_values('time')
        
        # O MERGE MÁGICO
        # direction='backward': Procura a transação que aconteceu ANTES ou NA HORA da oferta
        X_enriched = pd.merge_asof(
            X_copy,
            self.history_features,
            on='time',
            by='costumer_id',
            direction='backward',
            allow_exact_matches=False # False evita vazamento se a compra for o gatilho da oferta imediata
        )
        
        # Tratamento de Nulos (Para novos clientes sem histórico anterior)
        # Se não tem histórico, média de tempo pode ser -1 (indicador) e gasto é 0
        fill_values = {
            'avg_inter_purchase_hours': -1,
            'total_spent_so_far': 0,
            'trans_count_so_far': 0
        }
        X_enriched = X_enriched.fillna(fill_values)
        
        # Restaura a ordem original dos dados
        X_enriched = X_enriched.sort_values('_orig_index').drop(columns=['_orig_index'])
        
        return X_enriched