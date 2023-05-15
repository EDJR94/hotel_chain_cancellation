import pickle
import pandas as pd
import numpy as np

class HotelCancellation:
    def __init__(self):
        #self.home_path                = 'C:/Users/edils/repos/hotel_chain_cancelation/'
        
        #self.hospedes_scaler          = pickle.load(open(self.home_path + 'src/features/hospedes_scaler.pkl', 'rb'))
        #self.id_scaler                = pickle.load(open(self.home_path + 'src/features/id_scaler.pkl', 'rb'))        
        #self.meses_ate_checkin_scaler = pickle.load(open(self.home_path + 'src/features/meses_ate_checkin_scaler.pkl', 'rb'))        
        #self.nacionalidade_scaler     = pickle.load(open(self.home_path + 'src/features/nacionalidade_scaler.pkl', 'rb'))        
        #self.pernoites_scaler         = pickle.load(open(self.home_path + 'src/features/pernoites_scaler.pkl', 'rb'))        
        #self.tipo_quarto_scaler       = pickle.load(open(self.home_path + 'src/features/tipo_quarto_scaler.pkl', 'rb'))
         
        self.home_path                = ''
        
        self.hospedes_scaler          = pickle.load(open(self.home_path + 'features/hospedes_scaler.pkl', 'rb'))
        self.id_scaler                = pickle.load(open(self.home_path + 'features/id_scaler.pkl', 'rb'))        
        self.meses_ate_checkin_scaler = pickle.load(open(self.home_path + 'features/meses_ate_checkin_scaler.pkl', 'rb'))        
        self.nacionalidade_scaler     = pickle.load(open(self.home_path + 'features/nacionalidade_scaler.pkl', 'rb'))        
        self.pernoites_scaler         = pickle.load(open(self.home_path + 'features/pernoites_scaler.pkl', 'rb'))        
        self.tipo_quarto_scaler       = pickle.load(open(self.home_path + 'features/tipo_quarto_scaler.pkl', 'rb')) 
        
    def data_description(self, df1):
        valores_dict = ['id',
                    'classificacao', 
                    'meses_ate_checkin', 
                    'pernoites', 
                    'hospedes', 
                    'regime_alimentacao',
                    'nacionalidade', 
                    'forma_reserva', 
                    'ja_hospedou',
                    'tipo_quarto',
                    'reserva_agencia',
                    'reserva_empresa',
                    'reserva_estacionamento',
                    'reserva_observacoes']

        chaves_dict = df1.columns
        cols_new = dict(zip(chaves_dict,valores_dict))
        df1 = df1.rename(columns=cols_new)
        
        df1['hospedes'] = df1['hospedes'].fillna(2)
        df1['nacionalidade'] = df1['nacionalidade'].fillna('Spain')
    
        df1['hospedes'] = df1['hospedes'].astype('int64')
        
        return df1
    
    def data_transformation(self, df5):

        df5['meses_ate_checkin'] = self.meses_ate_checkin_scaler.transform(df5['meses_ate_checkin'].values.reshape(-1,1))
        #pickle.dump( rb, open ('C:/Users/edils/repos/hotel_chain_cancellation/src/features/meses_ate_checkin_scaler.pkl', 'wb'))

        df5['pernoites'] = self.pernoites_scaler.transform(df5['pernoites'].values.reshape(-1,1))
        #pickle.dump( rb, open ('C:/Users/edils/repos/hotel_chain_cancellation/src/features/pernoites_scaler.pkl', 'wb'))

        df5['hospedes'] = self.hospedes_scaler.transform(df5['hospedes'].values.reshape(-1,1))
        #pickle.dump( rb, open ('C:/Users/edils/repos/hotel_chain_cancellation/src/features/hospedes_scaler.pkl', 'wb'))


        #MinMax nos que tem distribuição normal
        df5['id'] = self.id_scaler.transform(df5['id'].values.reshape(-1,1))
        #pickle.dump( mms, open ('C:/Users/edils/repos/hotel_chain_cancellation/src/features/id_scaler.pkl', 'wb'))

        return df5

    def data_encoding(self, df5):

        #reserva_empresa - Label Encoder
        df5['reserva_empresa'] = df5['reserva_empresa'].apply(lambda x: 1 if x == 'Sim' else 0)

        #reserva_estacionamento - Label Encoder
        df5['reserva_estacionamento'] = df5['reserva_estacionamento'].apply(lambda x: 1 if x == 'Sim' else 0)

        #reserva_estacionamento - Label Encoder
        df5['reserva_agencia'] = df5['reserva_agencia'].apply(lambda x: 1 if x == 'Sim' else 0)

        #ja_hospedou - Label Encoder
        df5['ja_hospedou'] = df5['ja_hospedou'].apply(lambda x: 1 if x == 'Sim' else 0)  

        df5['classificacao'] = df5['classificacao'].apply(lambda x: x[0])
        df5['classificacao'] = df5['classificacao'].astype('int64')

        #One Hot Encoding - regime_alimentacao
        df5 = pd.get_dummies(df5, prefix='regime_alimentacao', columns=['regime_alimentacao'], dtype=int)

        #One Hot Encoding - reserva_observacoes
        df5 = pd.get_dummies(df5, prefix='reserva_observacoes', columns=['reserva_observacoes'], dtype=int)

        #One Hot Encoding - forma_reserva
        df5 = pd.get_dummies(df5, prefix='forma_reserva', columns=['forma_reserva'], dtype=int)

        df5 = df5.rename(columns={'regime_alimentacao_Café da manha':'cafe_manha',
                           'regime_alimentacao_Café da manha e jantar':'cafe_jantar',
                           'regime_alimentacao_Café da manha, almoco e jantar': 'cafe_almoco_jantar',
                           'regime_alimentacao_Sem refeicao': 'sem_refeicao',
                           'reserva_observacoes_1 a 3': 'obs_1_a_3',
                           'reserva_observacoes_Mais de 3': 'obs_mais_3',
                           'reserva_observacoes_Nenhuma': 'obs_nenhuma',
                           'forma_reserva_Agência': 'forma_reserva_agencia',
                           'forma_reserva_B2B': 'forma_reserva_b2b',
                           'forma_reserva_Balcão': 'forma_reserva_balcao'})
        #nacionalidade - Frequency Encoding
        df5['nacionalidade'] = df5['nacionalidade'].map(self.nacionalidade_scaler)
        #pickle.dump( values_nacionalidade, open ('C:/Users/edils/repos/hotel_chain_cancellation/src/features/nacionalidade_scaler.pkl', 'wb'))

        #tipo_quarto - Frequency Encoding
        values_quarto = df5['tipo_quarto'].value_counts(normalize=True)
        df5['tipo_quarto'] = df5['tipo_quarto'].map(self.tipo_quarto_scaler)
        #pickle.dump( values_quarto, open ('C:/Users/edils/repos/hotel_chain_cancellation/src/features/tipo_quarto_scaler.pkl', 'wb'))

        cols_selected_tree = ['id', 'nacionalidade', 'meses_ate_checkin', 'pernoites',
           'classificacao', 'obs_nenhuma', 'reserva_estacionamento', 'obs_1_a_3',
           'hospedes', 'forma_reserva_agencia', 'tipo_quarto','forma_reserva_balcao']
    
        return df5[cols_selected_tree]

    def get_prediction(self, model, original_data, test_data):
        
        yhat = model.predict(test_data)
        
        original_data['prediction'] = yhat
        
        return original_data.to_json(orient='records', date_format='iso')