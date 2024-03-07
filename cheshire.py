import aiml
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import text2emotion as te

def IsNaN(any_):
    return any_ != any_

def FixAIMLTimeError():
    import time
    time.clock = time.time

def FixText2EmotionEmojiError():
    import emoji
    emoji.UNICODE_EMOJI = emoji.EMOJI_DATA

def CreateAimlKernel(std_startup_path_, bot_name_):
    aimlKernel = aiml.Kernel()
    aimlKernel.learn(std_startup_path_)
    aimlKernel.respond("LOAD AIML B")
    aimlKernel.setBotPredicate("bot name", bot_name_)
    return aimlKernel

def GetAIMLOptionData(aiml_options_path_):
    df = pd.read_csv(aiml_options_path_)
    lsaData = np.array([i.replace('*', 'Cheshire') for i in df["pattern"].drop_duplicates().to_numpy()])
    emotionsOptionData = df[["template", "Happy", "Angry", "Surprise", "Sad", "Fear"]].drop_duplicates()
    return lsaData, emotionsOptionData

def GetPersonalityType():
    return str(input("personality type: "))

def GetUserInput():
    userInput = str(input("user: "))
    #remove special characters and then upercase all characters
    userInput = userInput.replace("[^a-zA-Z#]", " ").upper()
    return userInput

def GetAIMLOutput(aiml_kernel_, user_input_):
    return str(aiml_kernel_.respond(user_input_))

def GetAIMLOutputWithLSA(lsa_data_, aiml_kernel_, user_input_):
    lsaDataAndUserInput = np.append(lsa_data_, user_input_)
    print(lsaDataAndUserInput)
    vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
    X = vectorizer.fit_transform(lsaDataAndUserInput)
    svdModel = TruncatedSVD(n_components=len(lsaDataAndUserInput)-1, algorithm='randomized', n_iter=100, random_state=122)
    lsa = svdModel.fit_transform(X)
    similarityMatrix = cosine_similarity(lsa)
    largeestNumberIndex = np.argsort(similarityMatrix[len(lsaDataAndUserInput)-1])[-2]
    print(similarityMatrix[len(lsaDataAndUserInput)-1])
    print(np.argsort(similarityMatrix[len(lsaDataAndUserInput)-1]))
    print("new lsa input: ", lsaDataAndUserInput[largeestNumberIndex])
    return aiml_kernel_.respond(lsaDataAndUserInput[largeestNumberIndex])

def GetUserEmotion(user_input_):
    userEmotions = te.get_emotion(user_input_)
    if userEmotions == None:
        return None, 
    else:
        return max(zip(userEmotions.values(), userEmotions.keys()))[1]

def predict_emotion(personality_type, emotion_key):
    # define a dictionary that maps each MBTI type to a list of associated emotions
    mbti_emotions = {
        "ISTJ": ["Sad", "Fear", "Angry"],
        "ISFJ": ["Sad", "Surprise", "Fear"],
        "INFJ": ["Sad", "Fear", "Surprise"],
        "INTJ": ["Sad", "Surprise", "Angry"],
        "ISTP": ["Happy", "Surprise"],
        "ISFP": ["Sad", "Happy", "Surprise"],
        "INFP": ["Sad", "Happy", "Surprise"],
        "INTP": ["Surprise", "Fear", "Angry"],
        "ESTP": ["Happy", "Surprise"],
        "ESFP": ["Happy", "Surprise", "Fear"],
        "ENFP": ["Happy", "Surprise"],
        "ENTP": ["Surprise", "Fear", "Angry"],
        "ESTJ": ["Angry", "Fear", "Surprise"],
        "ESFJ": ["Happy", "Surprise", "Fear"],
        "ENFJ": ["Happy", "Surprise", "Fear"],
        "ENTJ": ["Angry", "Surprise", "Fear"]
    }
    
    # define a dictionary that maps each Enneagram type to a list of associated emotions
    enneagram_emotions = {
        "1": ["Sad", "Angry"],
        "2": ["Happy", "Surprise"],
        "3": ["Happy", "Surprise", "Angry"],
        "4": ["Sad", "Fear", "Angry"],
        "5": ["Surprise", "Fear"],
        "6": ["Fear", "Angry", "Surprise"],
        "7": ["Happy", "Surprise", "Fear"],
        "8": ["Angry", "Fear", "Surprise"],
        "9": ["Sad", "Happy", "Fear"]
    }
    
    # get user input for the personality type and the emotion key
    #personality_type = input("Enter your personality type (MBTI or Enneagram): ")
    #emotion_key = input("Enter an emotion (Sad, Happy, Fear, Surprise, Anger, Love, Disgust): ")
    
    # check if the personality type is valid and get the associated emotion list
    if personality_type in mbti_emotions.keys():
        emotion_list = mbti_emotions[personality_type]
    elif personality_type in enneagram_emotions.keys():
        emotion_list = enneagram_emotions[personality_type]
    else:
        raise ValueError("Invalid personality type!")
    
    # check if the emotion key is valid and return the associated emotion
    if emotion_key in emotion_list:
        return emotion_key
    else:
        # if the given emotion key is not in the list of associated emotions,
        # return the first emotion in the list (as a default)
        #if emotion_key not in ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']:
        if emotion_key not in ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']:
            #print("Invalid emotion key. Using default emotion.")
            return "template"
        return emotion_list[0]

def GetPersonalityOutput(emotions_option_data_, bot_emotion_, aiml_output_):
    templateIndex = emotions_option_data_.index[emotions_option_data_["template"] == aiml_output_].tolist()[0]
    if IsNaN(emotions_option_data_[bot_emotion_][templateIndex]):
        return emotions_option_data_["template"][templateIndex]
    else:
        return emotions_option_data_[bot_emotion_][templateIndex]

def main():
    FixAIMLTimeError()
    FixText2EmotionEmojiError()
    aimlKernel = CreateAimlKernel("std-startup.xml", "Cheshire")
    lsaData, emotionsOptionData = GetAIMLOptionData("aiml_options.csv")
    personalityType = GetPersonalityType()

    while True:
        userInput = GetUserInput()
        try: aimlOutput = GetAIMLOutput(aimlKernel, userInput)
        except: aimlOutput == ""
        if aimlOutput == "":
            aimlOutput = GetAIMLOutputWithLSA(lsaData, aimlKernel, userInput)
        userEmotion = GetUserEmotion(userInput)
        print("Emotion of user: ", userEmotion)
        botEmotion = predict_emotion(personalityType, userEmotion)
        personalityOutput = GetPersonalityOutput(emotionsOptionData, botEmotion, aimlOutput)
        print(f"bot:  {personalityOutput}")

if __name__ == '__main__': main()