import pandas as pd
from supervised import AutoML
import numpy as np

train_df = pd.read_csv('/home/woody/iwso/iwso092h/kaggle/p3e24_smoker_status/train.csv')
test_df = pd.read_csv('/home/woody/iwso/iwso092h/kaggle/p3e24_smoker_status/test.csv')
# sub_df = pd.read_csv('sample_submission.csv')

automl = AutoML(mode='Compete', eval_metric='auc')

X_df = train_df.drop('smoking', axis=1)
y_df = train_df.smoking

automl.fit(X_df, y_df)

preds = automl.predict_proba(test_df)
np.save('/home/woody/iwso/iwso092h/kaggle/p3e24_smoker_status/preds.npy', np.array(preds))


submission = pd.DataFrame()
submission['id'] = test_df.id
submission['smoking'] = preds

submission.to_csv('/home/woody/iwso/iwso092h/kaggle/p3e24_smoker_status/submission_1.csv', index=False)