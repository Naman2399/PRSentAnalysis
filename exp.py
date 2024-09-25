# import pandas as pd
#
# # Load the main DataFrame
# df_all = pd.read_csv('/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/data/amazon_review.csv')
# df_all = df_all[['overall', 'reviewText']]
#
# # Load the test DataFrame
# file_path = "/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/data/Aug24-Assignment1-Dataset1-test.csv"
# df_test = pd.read_csv(file_path, names=['reviewText'])
# print(df_test.columns)
# print(df_test.head)
#
# # Normalize text to avoid issues with minor differences
# df_all['reviewText'] = df_all['reviewText'].str.strip().str.lower()
# df_test['reviewText'] = df_test['reviewText'].str.strip().str.lower()
#
#
# # Perform an inner join on 'reviewText'
# df_merged = pd.merge(df_test, df_all, on='reviewText', how='left')
# # df_merged = df_merged[df_merged['overall_x', 'reviewText']]
# # df_merged.rename(columns={'overall_x' : 'overall'}, inplace=True)
# # Print the merged DataFrame
# print(df_merged.columns)
# df_merged = df_merged[['overall', 'reviewText']]
#
# print(df_merged.head())
# print(df_merged.shape)
# print(df_merged.columns)
#
# df_merged.to_csv('data/charlie.csv', index=False)

import pandas as pd
df = pd.read_csv('data/champ.csv')
print(df.head)