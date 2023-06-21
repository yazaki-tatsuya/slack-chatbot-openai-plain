def get_env_variable(key):

    env_variable_dict = {
        # ------------------------------------
        # OpenAI
        # ------------------------------------
        "OPEN_AI_KEY" : "",

        # ------------------------------------
        # App名：Slack_Python_Flask
        # ------------------------------------
        # BotユーザーID
        "BOT_USER_ID" : "",
        # Botトークン（Flask）
        "WEBAPPS_SLACK_TOKEN" : "",
        "WEBAPPS_SIGNING_SECRET" : "",

        # Botトークン（ソケットモード）
        "SOCK_SLACK_BOT_TOKEN" : "",
        "SOCK_SLACK_APP_TOKEN" : ""
    }
    ret_val = env_variable_dict.get(key, None)
    return ret_val