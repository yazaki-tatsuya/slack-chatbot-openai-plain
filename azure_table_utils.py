from azure.cosmosdb.table.tableservice import TableService
# 環境変数読み込み
import env

class AzureTableStorageUtils:
    
    def __init__(self, storage_name=None, storage_key=None):
        self.storage_name = storage_name
        self.storage_key = storage_key
        if self.storage_name is None:    
            self.storage_name = env.get_env_variable("AZURE_STORAGE_NAME")
        if self.storage_key is None:
            self.storage_key = env.get_env_variable("AZURE_STORAGE_KEY")

        self.tableService = TableService(account_name=self.storage_name, account_key=self.storage_key)

    def get_all_entities(self, table_name:str):
        query_result = self.tableService.query_entities(table_name) 
        if len(query_result.items) == 0:
            return None        
        else:
            return query_result

    def get_entities(self, table_name:str, filter:str=None, **kwargs):
        query_result = self.tableService.query_entities(table_name, filter=filter, **kwargs)
    
        if len(query_result.items) == 0:
            return None
        
        return query_result
    
    def insert_or_replace_entity(self, table_name:str, params:dict):
        result = self.tableService.insert_or_replace_entity(table_name, params)
        return result
    
    def merge_entity(self, table_name:str, params, if_match='*', timeout=None):
        result = self.tableService.merge_entity(table_name=table_name, entity=params, if_match=if_match, timeout=timeout)
        return result
