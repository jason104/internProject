import pandas as pd


df2 = pd.read_excel('e2.xlsx')
df3 = pd.read_excel('e3.xlsx')
df4 = pd.read_excel('e4.xlsx')
df5 = pd.read_excel('e5.xlsx')
df6 = pd.read_excel('e6.xlsx')

w = pd.ExcelWriter('final.xlsx', mode='a', if_sheet_exists='overlay')
df2.to_excel(w)
df3.to_excel(w)
df4.to_excel(w)
df5.to_excel(w)
df6.to_excel(w)
