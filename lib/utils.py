from lib.essay import Essay
from collections import defaultdict


def level_formatter(level):
    return f"""{{"level": {level}}}"""


def convert_essay(essays: list[Essay], system_message=None):
    # Initializing the messages list
    messages = []

    # Including the system message if provided
    if system_message:
        messages.append({
            "role": "system",
            "content": system_message
        })
    # Iterating through the lines and formatting the messages
    for essay in essays:
        # Formatting the message
        message = {
            "role": "user",
            "content": essay.text
        }
        messages.append(message)
        message = {
            "role": "assistant",
            "content": level_formatter(essay.level.value)
        }
        messages.append(message)

    # Creating the final output dictionary
    output_dict = {
        "messages": messages
    }
    return output_dict


def convert_conversation_samatha(conversation_str, system_message=None):
    conversation_str = conversation_str['conversation']
    # Splitting the conversation string into individual lines
    lines = conversation_str.split('\n\n')

    # Initializing the messages list
    messages = []

    # Including the system message if provided
    if system_message:
        messages.append({
            "role": "system",
            "content": system_message
        })

    # Iterating through the lines and formatting the messages
    for line in lines:
        # Splitting each line by the colon character to separate the speaker and content
        parts = line.split(': ', 1)
        if len(parts) < 2:
            continue

        # Identifying the role based on the speaker's name
        role = "user" if parts[0].strip() == "Theodore" else "assistant"

        # Formatting the message
        message = {
            "role": role,
            "content": parts[1].strip()
        }
        messages.append(message)

    # Creating the final output dictionary
    output_dict = {
        "messages": messages
    }

    return output_dict



def format_check(dataset):    
    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            if not content or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1
    return format_errors


# Yield successive n-sized 
# chunks from l. 
def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
