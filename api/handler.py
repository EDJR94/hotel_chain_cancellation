import pickle
import os
import pandas as pd
from flask import Flask, request, Response
from hotelcancellation.HotelCancellation import HotelCancellation

#loading model - Usar para Maquina Local
#path = 'C:/Users/edils/repos/hotel_chain_cancelation/'
#model = pickle.load(open(path + 'src/models/model_xgb.pkl','rb'))

#loading model - Usar para Deploy
model = pickle.load(open('models/model_xgb.pkl','rb'))

#Initiate API
app = Flask( __name__ )

#usar para Local
#@app.route( '/hotelcancellation/predict', methods=['POST'] )

@app.route( '/hotelcancellation/predict', methods=['POST'] )

def hotelcancellation_predict():
    
    test_json = request.get_json() #Load json for production
    
    if test_json: #there is data
        if isinstance(test_json, dict ): #only one row
            test_raw = pd.DataFrame(test_json, index=[0]) #keys for dict transform in columns
        
        else: #has multiple rows
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
    
        #Instatiate Hotel Class
        pipeline = HotelCancellation()

        #Data Description
        df1 = pipeline.data_description(test_raw)

        #Data Transformation
        df2 = pipeline.data_transformation(df1)

        #Data Encoding
        df3 = pipeline.data_encoding(df2)

        #prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response
    
    else:
        return Response('{}', status=200, mimetype='application/json')

#Usar isso para testar na maquina
#if __name__ == '__main__':
    #app.run('0.0.0.0', debug=True)

#Usar essa para Deploy
if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)    