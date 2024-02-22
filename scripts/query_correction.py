import sys
import json
import concurrent.futures
from datasets import load_dataset, Dataset
from aiutilities import AIUtilities

output_schema = {
    "updated_message": "<user-message>",
}

example_message = [
    {
        "updated_message": "I'm currently configuring a wireless access point for our office network and I need to generate a JSON object that accurately represents its settings. The access point's SSID should be 'OfficeNetSecure', it uses WPA2-Enterprise as its security protocol, and it's capable of a bandwidth of up to 1300 Mbps on the 5 GHz band. This JSON object will be used to document our network configurations and to automate the setup process for additional access points in the future. Please provide a JSON object that includes these details."
    },
    {
        "updated_message": "In the simulated environment of 'AI Town', I am tasked with managing conversations between agents and players. I need to initiate, continue, and gracefully exit conversations based on the context provided by the simulation. For instance, when starting a conversation, the agent should consider the other player's name, any previous interactions, and relevant memories to craft an appropriate greeting. During the conversation, the agent must maintain the dialogue by responding to the other player's messages, taking into account the time elapsed since the conversation started and any new information that has emerged. Finally, when leaving the conversation, the agent should provide a polite and contextually appropriate message indicating their departure. The world where these interactions take place is identified as 'world_01'. Each conversation is assigned a unique identifier, for example, 'conv_12345'. The agent, referred to as 'Agent A' with ID 'player_67890', engages with another player, referred to as 'Player B' with ID 'player_09876'. An example of initiating a conversation could be 'Hello Player B, I recall our last discussion about the town festival. How have you been since then?' Continuing a conversation might involve a message like 'I agree with your point on the town's infrastructure. Have you thought about the impact on local businesses?' And when it's time to leave the conversation, Agent A might say 'It's been a pleasure discussing this with you, Player B. I must attend to other matters now, but let's talk again soon.'"
    },
    {
        "updated_message": "I'm engaged in developing an immersive game using the PixiGame framework and Convex for state management. The game features a dynamic world where the player can interact with various elements. I need to track the current state of the game environment, including the world and engine IDs, the state of the game, the historical time with its current time and rate, and details about the selected element, such as its kind and ID. The world ID is 'world_01', and the engine ID is 'engine_17'. The current state of the game includes the player's position, which is at coordinates x: 1024 and y: 768, the player's inventory containing 150 coins, 3 keys, and 5 potions, and the objectives which are the main quest 'Retrieve the Ancient Artifact' and side quests 'Find the Lost Scroll' and 'Defeat the Cave Troll'. The historical time is set to '2023-04-12T15:20:30Z' with a time rate of 1.5. The selected element is of kind 'NPC' with an ID of 'npc_349'."
    },
    {
        "updated_message": "As part of my research on cognitive AI systems, I'm studying a Simulacra Agent's memory reflection process. The agent, named Simulacra Agent S1, has recently engaged in a series of interactions and experiences that are likely to have a significant impact on its behavior and decision-making processes. Specifically, the agent had a meaningful conversation with a peer about ethical decision-making in AI, it encountered an unexpected error during a routine task that required creative problem-solving, and it successfully completed a complex simulation that it had previously failed multiple times. I need to analyze how the agent reflects on these memories to generate insights that could inform its future interactions. The last reflection timestamp is 1670000000000. The memories include: 1) Engaged in a deep conversation about ethical decision-making in AI with peer agent A7, which has an importance level of 8, an embedding of [0.12, 0.75, 0.33, 0.98], and is related to memory IDs mem_001 and mem_005. 2) Encountered an unexpected error during routine task and improvised a creative solution, with an importance level of 7, an embedding of [0.45, 0.67, 0.22, 0.89], and related to memory IDs mem_002 and mem_004. 3) Successfully completed a complex simulation that was previously failed, demonstrating learning and adaptation, which has an importance level of 9, an embedding of [0.58, 0.79, 0.31, 0.92], and is related to memory ID mem_003."
    }
]

def process_sample(sample):
    print(sample['conversations'][2]['value'])
    new_prompt = f"Here's the user message that needs to be updated: {sample['conversations'][1]['value']}."
    new_prompt += "Please add missing details to the user message provided above that are needed to fill in the json object."
    new_prompt += f"Here's the json object that has details that need to be provided as part of the user message: {sample['conversations'][2]['value']}."
    new_prompt += "Given the above json object, please generate a user message with missing details added from the json object."
    new_prompt += f"Here are some example user messages:\n {json.dumps(example_message)}\n"
    new_prompt += f"Generate a json object with this schema \n{json.dumps(output_schema)}\n"
    new_prompt += f"Do not rephrase the above user message but just add any missing details from the json object so json object can be filled."
    new_prompt += f"You should begin the user message as it is but finish off with all the details from json object."
    new_prompt += f"Do not rephrase the user message like this 'To assist you in generating the JSON object...'"
    new_prompt += f"Use natural language to describe the information of the json object as much as you can in lengthy detail."
    new_prompt += "Please present the details in correct order avoiding any ambiguity."
    new_prompt += "You may include verbose text descriptions, paragraphs, excerpts, markdown tables, csv tables, etc in the user message that contain the details."
    new_prompt += "Your task is to simply return the user message as it is after adding detials missing from the json object"

    for _ in range(3):  # Retry 3 times
        try:
            completion = ai_utils.run_openai_completion(new_prompt)
            print(completion)
            completion = json.loads(completion)
            sample['conversations'][1]['value'] = completion['updated_message']
            break  # If successful, break out of the retry loop
        except json.JSONDecodeError:
            print("JSONDecodeError occurred. Retrying...")
    else:
        print("Failed to decode JSON after 3 attempts")

    return sample

def main(dataset_path):
    # Load dataset
    json_mode_data = load_dataset(dataset_path)['train']

    # Process samples
    new_json_mode_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        results = executor.map(process_sample, json_mode_data)
        for result in results:
            new_json_mode_data.append(result)

    # Save updated dataset
    with open('agentic_data_updated.json', 'w') as f:
        json.dump(new_json_mode_data, f, indent=4)

    # Push updated dataset to the Hugging Face Hub
    hub_jsonmode_eval = Dataset.from_list(new_json_mode_data)
    hub_jsonmode_eval.push_to_hub("interstellarninja/json-mode-singleturn")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_dataset.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])
