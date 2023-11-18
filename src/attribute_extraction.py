from google.cloud import aiplatform
import pandas as pd
import json
from pandas import json_normalize
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain, SQLDatabase
from google.oauth2 import service_account
from google.cloud import bigquery as bq

project_id = 'charged-formula-405300'
dataset_id = 'rag_llm'
table_id = 'attributes'

aiplatform.init(
    project=f'{project_id}',
    location='us-east1'
)

keyfile_path = 'charged-formula-405300-4a513f476167.json'

credentials = service_account.Credentials.from_service_account_file(
    keyfile_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
bq_client = bq.Client(credentials = credentials, project= f'{project_id}')

def query_llm_with_one_argument(query_template, argument):
    prompt = PromptTemplate(template=query_template, input_variables=['input'])

    llm = VertexAI()
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain.run(argument)


def attribute_extraction(df):
    df_dict = df[['Review', 'Rating']].iloc[0:1000,:].to_dict()
    product_reviews = []

    for i in df_dict['Review'].values():
        product_reviews.append(i)

    extraction_template = """
    Your goal is to extract structured information from the user's review that matches the form described below. When extracting information please make sure it matches the type information exactly. Do not add any attributes that do not appear in the schema shown below.
    
    ```TypeScript
    review: {{                 // Review attributes
       sentiment: string            // Positive or Negative or Neutral, Sentiment of the review
       liked: string         // What user liked about the product - comma separated phrases in 2 words
       disliked: string      // What user disliked about the product - comma separated phrases in 2 words
    }}
    ```
    
    Please output the extracted information in JSON format. Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional fields that do not appear in the schema.
    
    Input: I have been using Halo for a couple of months now and there is so much to like but it feels a bit rough around the edges when compared to other more refined products from Fitbit or Samsung. I have had a number of smart watches and fitness bands over the years. I have been using a high end smart watch for the last year. The problem with that is that I cannot wear my good analog watches without looking like an idiot. Even the Fitbit fitness bands have a large digital display which looks somewhat stupid when you have a big dive watch on the other wrist. So, I wanted something small and really understated. Amazon had invited me to try Halo (paid for it but at a discount) and it looked like it might fit the bill. For the most part it does. Here are the highlights:-Hardware: Not having a screen is strange at first but I have learned to really like it. With all of our devices, it is one less thing to distract me. The device itself is lightweight aluminum that fits the wrist nicely. It has one button for resetting the device if ever needed. A double tap will resync the app and the band if you have had both connected for a bit without using. There are the typical sensors on the bottom to measure your heart rate. I got one with the silver mesh strap. I really like it but opted to buy a silicone strap as the Halo is waterproof and I like to shower without removing it. The mesh strap stayed wet too long for my liking. Amazon has some work to do on the silicone strap designs. They are all pretty ugly but they work. All and all this is a very comfortable device that you won't mind wearing all day everyday.-Software: As one would expect from a company like Amazon, the software experience is where things shine for me. It is still incomplete in some ways but you can see where they are heading and the future is bright. As it sits, it is still pretty great. There are four tabs at the bottom of the app; Data, Lab, Live, and Settings. Data is where your various stats are displayed. Those include Activity, Sleep, Tone, and Body. Let's dig into these.-Activity: Amazon is taking a total activity approach to healthy living. Less about steps though they are there if you want to see them. The idea is that your activity levels throughout the day accumulate for points. For instance, I got 50 points for a 50 minute elliptical session this morning. It automatically picks up your activities but doesn't necessarily always know what you are doing. After my workouts I just go back in and edit them if they are not properly identified. The Halo starts you out at 150 points a week as a target and moves you up as you achieve your target points.  All and all it works great and keeps me motivated.-Sleep tracking: The sleep tracking is thorough and robust. Though if you are like me and sometimes fall asleep on the sofa for an hour and then go to bed, the Halo band will almost always fail to register the first hour. Seems like a simple update at some point. I like the point system Amazon employs here. It drives me to go to bed earlier and leave my TV off.-Tone: This one is a weird one. Basically, the Halo band listens to you and analyzes your tone of voice to try and measure stress and anxiety. It does actually work but I didn't really get much out of it honestly. Further, it significantly reduces your battery life. Maybe by as much as 60%. For me, I would rather have the battery life but I think this will be a highly individual experience. Some will really appreciate this new approach.-Body: Unless you are in really really good shape, this one will probably haunt you a bit but man is it effective so I really recommend utilizing this feature. Basically you set your phone down a solid ten or so feet away from you at waist height with just your underwear on. The app scans your body, captures the image, and calculates your body fat. It is fairly accurate. Maybe a little heavy handed. I am a fifty year old man that works out regularly. Let's just say, after seeing the image and the body fat percentage, I have a lot of work to do. I am extremely motivated to improve both the fat percentage and the image I am looking at in the app. I think the easiest thing to do in life is lie to yourself about how you look. This feature makes that impossible. It hurts but I really like this feature. I will absolutely get healthier because of it. I would add a word of caution here. If you are someone that has struggled with body issues in the past, you might not be right for this device. It definitely makes you more conscious of your body and appearance.The next tab at the bottom is lab. This is where I think Amazon's future in fitness has the most potential. They have partnered with dozens of companies to provide workouts, stretching, yoga, meditations, diet info, and more. I did an Orangetheory workout this morning. It was brutal. I also have been doing a two week bedtime guided meditation. It has been amazing. This is all built into your monthly subscription ($3.99) after your free trial. It really is great and provides an absolute ton of added content value. The best part is Amazon is just getting started here. With Amazon's market power, I have to imagine that Lab is going to become unbelievably valuable in the coming years. If you have been with Prime for a number of years, I can imagine a growth curve similar to that where Amazon just continues to add more value, more content, and probably more cost. I look forward to seeing this.The next tab is the Live tab. Live is simple but great and it is something I wish my Samsung Active 2 offered. You can hit the tab button and see your heart rate or your tone (mood) in real time. When I get on a treadmill, I fire up my jams, go to the Halo app, and hit the Live tab. It shows me an accurate measurement of my heart rate that updates continuously throughout my run. Its something like having a chest strap monitor on without having it on. It has helped me a great deal.Lastly, you have settings. All the typical things you might want to tweak are in there. Solid.All and all, the software experience here is very good and it is likely to get much better. Amazon gave Halo a big update not long ago, presumably in anticipation of it going on sale to the public. It received a lot of upgrades. If Amazon continues to develop and support Halo band, as I anticipate they will, this will just get better.-Battery Life / Charging: With tone turned off, I get in excess of ten days on a charge. With it on, its closer to five or six. Charging from zero-ish percent takes around two hours via a proprietary charger. It is big and I don't love it but it works. So, A for me on battery life and more like a C- on the charger itself.I will update this review as I go with more information but, for now, call me impressed. It isn't perfect. The device is a little clunky and unrefined. Tone is cool but weird. The body scan might bruise your ego a bit. I would love to see the addition of a calorie counting section on the app and a more robust tracking of actual exercise history. After all of that though, what Amazon is really focused on here are three really important aspects of your life: How active are you, how are you sleeping, and how fat you are. If we all just stayed focused on those things, the world would be a much healthier place. So, I am recommending giving Halo a shot, especially if you are sick of the connectivity a smart watch gives you. Today, it just works and motivates me. I cannot wait to see what the future holds with the Amazon Halo Band.***Update***One thing I forgot to add but is important relates to why the Amazon approach to activity is important. Most fitness bands / watches track steps. It is a solid metric. What makes Halo better is, while it tracks steps, it does so in conjunction with the effort you are putting in. So let me show you what I mean. Today, I got credit for just over 4,000 steps on my 50 minute elliptical workout. I got 51 points for it. After going about my day, showering, working, getting ready for tomorrow, etc. I have finished my day at just over 15,000 steps. However, I have only earned an additional 20 points to finish at 71 points for the day. The point is, that the Halo Band is able to weigh steps during a workout as being more valuable than steps taken running around the office or my home. You get credit for those steps but the weight is scored heavier when your heart rate is higher. Steps are good but it is the kind of steps you are taking that make the difference to your health and weight. There are other bands that do this kind of thing but I think that Amazon has executed it particularly well here. One other thing I should have added from my first go is that Halo is partnered with Weight Watchers. My wife uses Weight Watchers and she finds Halo to be incredibly valuable to her efforts. It automatically syncs your activities in your WW account so she doesn't need to track that manually any longer. Great addition for all you Weight Watchers fans. It is available through Labs.03/05/21 Update: Amazon recently released a substantial update to the band which does a number of useful things. First and foremost, it now has voice integration with Alexa that is useful. You can just ask for updates like ""Hey Alexa, how many steps do I have today?"" It is a nice integration. It is optional in the app. I will include a picture of the toggle above. Additionally, they have enhanced sleep tracking. It is more robust and allows you to tap your finger on your sleep grid at any point to see the time much like Fitbit does. They have also enhanced 'Discover' which allows you to more easily find new content such as workouts and meditations you might want to try. Lastly, they added a calendar sync for Tone which I do not use due to the afformentioned battery drain. However, this looks pretty neat. So, say you have ""lunch with Mom"" on your calendar. Halo will track your tone during that time and let you know how your tone and mood seemed during the lunch. I won't use it but it seems like a useful enhancement to a feature that is unique to the Halo band. This update is a great example of the potential this band has if Amazon continues to grow it and support it.
    Output: {{ "sentiment": "Positive", "liked": "Small size, Lightweight design, Comfortable to wear, No distracting screen, Comprehensive software experience, Thorough sleep tracking, Body composition assessment", "disliked": "Clunky and unrefined device, Weird Tone feature, Body scan might be intimidating, Charger design could be improved" }}
    
    Input: {input}
    Output:
    """

    dfs = []
    for i,j in enumerate(product_reviews):
        print(i)
        try:
            json_string = json.loads(query_llm_with_one_argument(extraction_template, j))
            flattened_data = json_normalize(json_string)
        except Exception as e:
            print(f'ignored {i}- comment and moving forward with other reviews')
        dfs.append(flattened_data)

    return pd.concat(dfs, ignore_index = True)

def push_df_to_bq(df):
    table = f'{project_id}.{dataset_id}.{table_id}'

    job_config = bq.LoadJobConfig()

    bq_client.load_table_from_dataframe(df, table, job_config=job_config)

if __name__ == '__main__':
    df = pd.read_csv('Smart_Watch_Review.csv')
    final_df = attribute_extraction(df)
    push_df_to_bq(final_df)