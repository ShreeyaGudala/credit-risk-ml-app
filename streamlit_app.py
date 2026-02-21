[     UTC     ] Logs for credit-risk-ml-app-zwhdgcvekos8ocaquot3bt.streamlit.app/

────────────────────────────────────────────────────────────────────────────────────────

[07:48:08] 🚀 Starting up repository: 'credit-risk-ml-app', branch: 'main', main module: 'streamlit_app.py'

[07:48:08] 🐙 Cloning repository...

[07:48:09] 🐙 Cloning into '/mount/src/credit-risk-ml-app'...

[07:48:09] 🐙 Cloned repository!

[07:48:09] 🐙 Pulling code changes from Github...

[07:48:09] 📦 Processing dependencies...


──────────────────────────────────────── uv ───────────────────────────────────────────


Using uv pip install.

Using Python 3.13.12 environment at /home/adminuser/venv

Resolved 49 packages in 490ms

Prepared 49 packages in 5.94s

Installed 49 packages in 104ms

 + altair==6.0.0

 + attrs==25.4.0

 + blinker==1.9.0

 + cachetools==6.2.6

 + certifi==2026.1.4

 + charset-normalizer==3.4.4

 + click==8.3.1

 + contourpy==1.3.3

 + cycler==0.12.1[2026-02-21 07:48:16.483759] 

 + fonttools==4.61.1

 + gitdb==4.0.12

 + gitpython==3.1.46

 + idna==3.11

 + jinja2==3.1.6

 + joblib==1.5.3

 + jsonschema==4.26.0

 + jsonschema-specifications==2025.9.1

 + kiwisolver==1.4.9

 + markupsafe==3.0.3

 + matplotlib==3.10.8

 + narwhals==[2026-02-21 07:48:16.484220] 2.16.0

 + numpy==2.4.2

 + nvidia-nccl-cu12==2.29.3

 + packaging==26.0

 + pandas==2.3.3

 + pillow==12.1.1

 + protobuf==6.33.5

 +[2026-02-21 07:48:16.484345]  pyarrow==23.0.1

 + pydeck==0.9.1

 + pyparsing==3.3.2

 + python-dateutil==2.9.0.post0

 + pytz==2025.2

 + referencing==0.37.0

 + requests[2026-02-21 07:48:16.484527] ==2.32.5

 + rpds-py==0.30.0

 + scikit-learn==1.8.0

 + scipy==1.17.0

 + six==1.17.0

 + smmap==5.0.2

 + streamlit==1.54.0

 + tenacity==9.1.4

 + threadpoolctl==3.6.0

 + toml==0.10.2

 + tornado[2026-02-21 07:48:16.484659] ==6.5.4

 + typing-extensions==4.15.0

 + tzdata==2025.3

 + urllib3==2.6.3

 + watchdog==6.0.0

 + xgboost==3.2.0

Checking if Streamlit is installed

Found Streamlit version 1.54.0 in the environment

Installing rich for an improved exception logging

Using uv pip install.

Using Python 3.13.12 environment at /home/adminuser/venv

Resolved 4 packages in 194ms

Prepared 4 packages in 181ms

Installed 4 packages in 56ms

 + markdown-it-py==4.0.0

 + mdurl[2026-02-21 07:48:18.350710] ==0.1.2

 + pygments==2.19.2

 + rich==14.3.3


────────────────────────────────────────────────────────────────────────────────────────


[07:48:19] 🐍 Python dependencies were installed from /mount/src/credit-risk-ml-app/requirements.txt using uv.

Check if streamlit is installed

Streamlit is already installed

[07:48:20] 📦 Processed dependencies!




[08:01:03] 🐙 Pulling code changes from Github...

[08:01:03] 📦 Processing dependencies...

[08:01:03] 📦 Processed dependencies!

[08:01:04] 🔄 Updated app!

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/credit-risk-ml-app/streamlit_app.py:71 in <module>                 

                                                                                

     68 │   │   "AMT_INCOME_TOTAL": [amt_income],                               

     69 │   │   "AMT_CREDIT": [amt_credit],                                     

     70 │   │   "AMT_ANNUITY": [amt_annuity],                                   

  ❱  71 │   │   "EXT_SOURCE_2": [ext_source_2],                                 

     72 │   │   "EXT_SOURCE_3": [ext_source_3]                                  

     73 │   })                                                                  

     74                                                                         

────────────────────────────────────────────────────────────────────────────────

NameError: name 'ext_source_2' is not defined

[08:05:35] 🐙 Pulling code changes from Github...

[08:05:36] 📦 Processing dependencies...

[08:05:36] 📦 Processed dependencies!

[08:05:37] 🔄 Updated app!

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/credit-risk-ml-app/streamlit_app.py:75 in <module>                 

                                                                                

     72 │   │   "credit_history_strength": [credit_history_strength]            

     73 │   })                                                                  

     74 │                                                                       

  ❱  75 │   probability = model.predict_proba(input_data)[0][1]                 

     76 │   st.session_state.probability = probability                          

     77                                                                         

     78 # -------------------------------                                       

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/xgboost/sklearn.py:1921 in  

  predict_proba                                                                 

                                                                                

    1918 │   │   │   )                                                          

    1919 │   │   │   class_prob = softmax(raw_predt, axis=1)                    

    1920 │   │   │   return class_prob                                          

  ❱ 1921 │   │   class_probs = super().predict(                                 

    1922 │   │   │   X=X,                                                       

    1923 │   │   │   validate_features=validate_features,                       

    1924 │   │   │   base_margin=base_margin,                                   

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py:751 in      

  inner_f                                                                       

                                                                                

     748 │   │   │   │   warnings.warn(msg, FutureWarning)                      

     749 │   │   │   for k, arg in zip(sig.parameters, args):                   

     750 │   │   │   │   kwargs[k] = arg                                        

  ❱  751 │   │   │   return func(**kwargs)                                      

     752 │   │                                                                  

     753 │   │   return inner_f                                                 

     754                                                                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/xgboost/sklearn.py:1446 in  

  predict                                                                       

                                                                                

    1443 │   │   │   iteration_range = self._get_iteration_range(iteration_ran  

    1444 │   │   │   if self._can_use_inplace_predict():                        

    1445 │   │   │   │   try:                                                   

  ❱ 1446 │   │   │   │   │   predts = self.get_booster().inplace_predict(       

    1447 │   │   │   │   │   │   data=X,                                        

    1448 │   │   │   │   │   │   iteration_range=iteration_range,               

    1449 │   │   │   │   │   │   predict_type="margin" if output_margin else "  

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py:751 in      

  inner_f                                                                       

                                                                                

     748 │   │   │   │   warnings.warn(msg, FutureWarning)                      

     749 │   │   │   for k, arg in zip(sig.parameters, args):                   

     750 │   │   │   │   kwargs[k] = arg                                        

  ❱  751 │   │   │   return func(**kwargs)                                      

     752 │   │                                                                  

     753 │   │   return inner_f                                                 

     754                                                                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py:2854 in     

  inplace_predict                                                               

                                                                                

    2851 │   │   if _is_pandas_df(data):                                        

    2852 │   │   │   data, fns, _ = _transform_pandas_df(data, enable_categori  

    2853 │   │   │   if validate_features:                                      

  ❱ 2854 │   │   │   │   self._validate_features(fns)                           

    2855 │   │   if _is_list(data) or _is_tuple(data):                          

    2856 │   │   │   data = np.array(data)                                      

    2857                                                                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py:3429 in     

  _validate_features                                                            

                                                                                

    3426 │   │   │   │   │   + ", ".join(str(s) for s in my_missing)            

    3427 │   │   │   │   )                                                      

    3428 │   │   │                                                              

  ❱ 3429 │   │   │   raise ValueError(msg.format(self.feature_names, feature_n  

    3430 │                                                                      

    3431 │   def get_split_value_histogram(                                     

    3432 │   │   self,                                                          

────────────────────────────────────────────────────────────────────────────────

ValueError: feature_names mismatch: ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 

'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] ['AMT_INCOME_TOTAL', 

'AMT_CREDIT', 'AMT_ANNUITY', 'repayment_reliability', 'credit_history_strength']

expected EXT_SOURCE_3, EXT_SOURCE_2 in input data

training data did not have the following fields: credit_history_strength, 

repayment_reliability

[08:16:44] 🐙 Pulling code changes from Github...

[08:16:45] 📦 Processing dependencies...

[08:16:45] 📦 Processed dependencies!

[08:16:46] 🔄 Updated app!

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/credit-risk-ml-app/streamlit_app.py:73 in <module>                 

                                                                                

     70 │   "AMT_ANNUITY": [amt_annuity],                                       

     71 │   "EXT_SOURCE_2": [repayment_reliability],                            

     72 │   "EXT_SOURCE_3": [credit_history_strength],                          

  ❱  73 │   "CREDIT_TO_ANNUITY_RATIO": [credit_to_annuity_ratio]                

     74 })                                                                      

     75 │                                                                       

     76 │   probability = model.predict_proba(input_data)[0][1]                 

────────────────────────────────────────────────────────────────────────────────

NameError: name 'credit_to_annuity_ratio' is not defined

[08:19:21] 🐙 Pulling code changes from Github...

[08:19:22] 📦 Processing dependencies...

[08:19:22] 📦 Processed dependencies!

[08:19:23] 🔄 Updated app!

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/credit-risk-ml-app/streamlit_app.py:78 in <module>                 

                                                                                

     75 │   │   "CREDIT_TO_ANNUITY_RATIO": [credit_to_annuity_ratio]            

     76 │   })                                                                  

     77 │                                                                       

  ❱  78 │   probability = model.predict_proba(input_data)[0][1]                 

     79 │   st.session_state.probability = probability                          

     80                                                                         

     81 # -------------------------------                                       

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/xgboost/sklearn.py:1921 in  

  predict_proba                                                                 

                                                                                

    1918 │   │   │   )                                                          

    1919 │   │   │   class_prob = softmax(raw_predt, axis=1)                    

    1920 │   │   │   return class_prob                                          

  ❱ 1921 │   │   class_probs = super().predict(                                 

    1922 │   │   │   X=X,                                                       

    1923 │   │   │   validate_features=validate_features,                       

    1924 │   │   │   base_margin=base_margin,                                   

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py:751 in      

  inner_f                                                                       

                                                                                

     748 │   │   │   │   warnings.warn(msg, FutureWarning)                      

     749 │   │   │   for k, arg in zip(sig.parameters, args):                   

     750 │   │   │   │   kwargs[k] = arg                                        

  ❱  751 │   │   │   return func(**kwargs)                                      

     752 │   │                                                                  

     753 │   │   return inner_f                                                 

     754                                                                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/xgboost/sklearn.py:1446 in  

  predict                                                                       

                                                                                

    1443 │   │   │   iteration_range = self._get_iteration_range(iteration_ran  

    1444 │   │   │   if self._can_use_inplace_predict():                        

    1445 │   │   │   │   try:                                                   

  ❱ 1446 │   │   │   │   │   predts = self.get_booster().inplace_predict(       

    1447 │   │   │   │   │   │   data=X,                                        

    1448 │   │   │   │   │   │   iteration_range=iteration_range,               

    1449 │   │   │   │   │   │   predict_type="margin" if output_margin else "  

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py:751 in      

  inner_f                                                                       

                                                                                

     748 │   │   │   │   warnings.warn(msg, FutureWarning)                      

     749 │   │   │   for k, arg in zip(sig.parameters, args):                   

     750 │   │   │   │   kwargs[k] = arg                                        

  ❱  751 │   │   │   return func(**kwargs)                                      

     752 │   │                                                                  

     753 │   │   return inner_f                                                 

     754                                                                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py:2854 in     

  inplace_predict                                                               

                                                                                

    2851 │   │   if _is_pandas_df(data):                                        

    2852 │   │   │   data, fns, _ = _transform_pandas_df(data, enable_categori  

    2853 │   │   │   if validate_features:                                      

  ❱ 2854 │   │   │   │   self._validate_features(fns)                           

    2855 │   │   if _is_list(data) or _is_tuple(data):                          

    2856 │   │   │   data = np.array(data)                                      

    2857                                                                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py:3429 in     

  _validate_features                                                            

                                                                                

    3426 │   │   │   │   │   + ", ".join(str(s) for s in my_missing)            

    3427 │   │   │   │   )                                                      

    3428 │   │   │                                                              

  ❱ 3429 │   │   │   raise ValueError(msg.format(self.feature_names, feature_n  

    3430 │                                                                      

    3431 │   def get_split_value_histogram(                                     

    3432 │   │   self,                                                          

────────────────────────────────────────────────────────────────────────────────

ValueError: feature_names mismatch: ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 

'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] ['AMT_INCOME_TOTAL', 

'AMT_CREDIT', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 

'CREDIT_TO_ANNUITY_RATIO']

training data did not have the following fields: CREDIT_TO_ANNUITY_RATIO

main
shreeyagudala/credit-risk-ml-app/main/streamlit_app.py
