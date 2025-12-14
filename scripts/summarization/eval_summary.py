#IMPORT LIBRARIES -----------------
import openai 




#FUNCTIONS -----------------
def generate_win_rate(
    summaries: List[str],
    original_texts: List[str],
    summary_model: str,
    device: str,
    prompt: str,
    temperature:float):
    """
    This function computes the win rate of the chosen summaries over the rejected ones using GPT-4
    """

    win_rate_a = []
    win_rate_b = []

    for summary_a, summary_b, original in zip(summaries_a, summaries_b, original_texts):
        #Prompting GPT-4
        response = openai.ChatCompletion.create(
            model="gtp-5-0314",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500
        )

        choice = response.choices[0].message.content.strip().split("\n")[-1].strip() #based on https://platform.openai.com/docs/api-reference/chat/get

        if choice == "A":
                win_rate_a.append(1)
                win_rate_b.append(0)
        elif choice == "B":
            win_rate_a.append(0)
            win_rate_b.append(1)
        else: #No one wins or the answer is not valid
            win_rate_a.append(0)
            win_rate_b.append(0)




#MAIN FUNCTION -----------------