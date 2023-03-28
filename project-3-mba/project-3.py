import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("anonymous-msweb.data", skiprows=301, header=None)
print(df.head())

current_id = None
progress_interval = int(len(df) / 60)
print("Processing...")
print("_" * 60)
for index, row in df.iterrows():
    if index != 0 and index % progress_interval == 0:
        print("▊", end="", flush=True)
    if row[0] == 'C':
        current_id = row[2]
        df.drop(index, inplace = True) #FIXME: slow
    else:
        df.at[index, 2] = current_id
print("\n")

df.rename(columns={0:'type', 1:'item', 2:'user'}, inplace = True)
df.drop('type', axis = 1, inplace = True)
print(df.head(10))

df.item = df.item.transform(lambda x: [x])
df = df.groupby(['user']).sum()['item'].reset_index(drop = True)

encoder = TransactionEncoder()
transactions = pd.DataFrame(encoder.fit(df).transform(df), columns=encoder.columns_)
print(transactions.head())

frequent_itemsets = apriori(transactions, min_support = 0.007, use_colnames = True)
rules = association_rules(frequent_itemsets, metric = "lift",  min_threshold = 1)
print(rules.head())
print(f"Total rules: {len(rules)}")

###### BREAK #####

names = pd.read_csv("anonymous-msweb.data", skiprows=7, nrows=293, header = None)

names.rename(columns = {0:'type', 1:'id', 2:'N', 3:'name', 4:'path'}, inplace = True)
names.drop(['type', 'N'], axis = 1, inplace = True)
# print(names.head(10))

query = ""
while query != "quit":
  query = input("Enter site name: ")
  if query == "quit": continue

  try:
    query_id = names.loc[names['path'] == query].iloc[0]['id']
    results = rules.loc[rules['antecedents'].apply(lambda x: query_id in x)]

    if results.empty:
      print("No recommendation")
    else:
      highest = results.loc[results['lift'].idxmax()]

      res = []
      for x in highest['consequents']:
        res.append(nameFromId(names, x))

      print(f"Recommendation: {', '.join(res)} (lift: {highest['lift']})")

  except IndexError:
    print(f"Invalid site name: {query}")
