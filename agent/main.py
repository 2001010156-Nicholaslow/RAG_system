# main.py
from react_agent import pass_llm_query, clarify_query
from routing_agent import route_query

user_roles = ["default",]

def handle_user_input(user_input: str):
    route_result = route_query(user_input, unload_after=True)
    print((f"Routing start: {route_result}"))
    if route_result["route"] == "agent":
        if route_result["confidence"] < 0.4:
            #print("Unclear")
            clarification = clarify_query(route_result["raw_input"])
            print("\n🤖 Clarification:", clarification)
        else:
            result = pass_llm_query(route_result, user_roles=user_roles)
            print("\n🤖 Agent Response:\n", result)
    elif route_result["route"] == "tool":
        # Call tool directly (optional)
        print("Tool-specific logic not implemented yet.")
    else:
        print("Sorry, I don't understand that yet.")

if __name__ == "__main__":
    user_input = input("You: ")
    handle_user_input(user_input)
