import json
from datetime import datetime, timedelta
#################################################
# SAMPLE (Function Callingで呼び出す関数を追記)
#################################################
# 出発地と目的地を引数としてフライト情報を取得する関数
def get_flight_info(departure, destination):
    """
    出発地と目的地の間のフライト情報を取得する関数
    """
    # デモのためのダミーのフライト情報（本来はDBやAPI経由で取得する）
    flight_info = {
        "departure": departure,
        "destination": destination,
        "datetime": str(datetime.now() + timedelta(hours=2)),
        "airline": "JAL",
        "flight": "JL0006",
    }
    # フライト情報をJSON形式で返す
    return json.dumps(flight_info)