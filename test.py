import requests
# import pandas as pd
# import numpy as np
postsession = requests.Session()
t = postsession.post('https://timbre-cx7hn2quva-uc.a.run.app/song', 
                     files={'voice': open('C:/Users/small/p.mp3', 'rb')})
print(t)

print(t.text)
print(t.json())

# attrval = {'hardness':[43.21439351,59.75435847],'depth':[24.29675547,58.40741266],'brightness':[52.26069264,71.37626087],'roughness':[24.66124888,57.98019528],'warmth':[33.21028496,54.25769172],'sharpness':[37.02482355,60.0009997],'boominess':[2.078721208,41.60153737]}
# attrs = ['hardness', 'depth', 'brightness',
#          'roughness', 'warmth', 'sharpness', 'boominess']
# def timbre_similarity(timbre1, timbre2):
#     similarity = 0
#     for attr in attrs:
#         if not np.isnan(pd.to_numeric(timbre2[attr],errors='coerce')):
#             similarity += (timbre1[attr] - pd.to_numeric(timbre2[attr])) ** 2
#     return similarity

# def timbre_similarity_list(input_vocal_path,genre,start,end,song = False):
#     timbre = {'boominess': 1.0063232935542983, 'brightness': 0.27651377712986114, 'depth': 0.9912161562552775, 'hardness': 0.5691462010041751, 'reverb': 0, 'roughness': 0.46902887200112325, 'sharpness': 0.07461452982296704, 'warmth': 0.6369530001005269}
#     or_timbre = timbre.copy()
#     if song:
#         database_path = 'C:/Users/small/OneDrive/바탕 화면/FeatureExtraction/crawling2.csv'
#     else:
#         database_path = 'crawling.csv'
#     dataframe = pd.read_csv(database_path)
#     dataframe['timbre_similarity'] = 0
#     for index in range(dataframe.shape[0]):
#         dataframe['timbre_similarity'][index] = timbre_similarity(
#             timbre, dict(dataframe[attrs].loc[index]))
#     for a in attrs:
#         minv = attrval[a][0]
#         maxv = attrval[a][1]
#         timbre[a] = (timbre[a] - minv) / (maxv - minv)
#     new_df = dataframe.sort_values(by=['timbre_similarity'])
#     new_df=new_df.replace(np.nan, '1')
#     print(new_df.release,new_df.genre)
#     masked =False
#     if type(genre) == list:
#         mask = None
#         for singlegen in genre:
#             #if new_df.genre.str.contains(singlegen, na=False) !=None:
#             if mask is None:
#                 mask = new_df.genre.str.contains(singlegen, na=False)
#             else:
#                 mask = mask | new_df.genre.str.contains(singlegen, na=False)
#         if start !=0 or end !=9:
#             mask = (mask) & (start< new_df.release) & (new_df.release<end)
#         masked = True
#     else:
#         mask = False
#         if genre:
#             mask = new_df.genre.str.contains(genre, na=False)
#             masked =True
#         if start !='0' or end !='9':
#             if genre:
#                 mask = mask & ((start< new_df.release) & (new_df.release<end))
#             else:
#                 mask = ((start< new_df.release) & (new_df.release<end))
#             masked = True
#     if masked:
#         res_df = new_df.loc[mask,:]
#     else:
#         res_df = new_df
#     print(res_df)

#     return or_timbre,timbre, res_df[['id', 'title', 'singer', 'genre', 'release', 'timbre_similarity']].head(10)