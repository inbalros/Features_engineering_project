import featuretools as ft
data = ft.demo.load_mock_customer()
customers_df = data["customers"]
sessions_df = data["sessions"]
transactions_df = data["transactions"]
entities = {
"customers" : (customers_df, "customer_id"),
"sessions" : (sessions_df, "session_id", "session_start"),
"transactions" : (transactions_df, "transaction_id", "transaction_time") }
relationships = [("sessions", "session_id", "transactions", "session_id"),
                 ("customers", "customer_id", "sessions", "customer_id")]

feature_matrix_customers, features_defs = ft.dfs(entities=entities, relationships=relationships,target_entity="customers")

print(feature_matrix_customers)
print(features_defs)
print(features_defs[1])
feature = features_defs[1]
#ft.graph_feature(feature)
print(ft.describe_feature(feature))



from featuretools.primitives import AddNumeric
add_numeric = AddNumeric()
print(add_numeric([2, 1, 2], [1, 2, 2]).tolist())

