from flask import Flask, render_template, request
import cheshire as bot

app = Flask(__name__, template_folder=".")
bot.FixAIMLTimeError()
bot.FixText2EmotionEmojiError()
aimlKernel = bot.CreateAimlKernel("std-startup.xml", "Cheshire")
lsaData, emotionsOptionData = bot.GetAIMLOptionData("aiml_options.csv")

@app.route('/')
def my_form():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def my_form_post():
    if request.method == 'POST':
        personalityType = request.form['selectPersonality']
        if personalityType == "":
            personalityType = "ISTJ"
        userInput = request.form['userInput']
        aimlOutput = bot.GetAIMLOutput(aimlKernel, userInput)
        #print("aiml: ", aimlOutput)
        if aimlOutput == "":
            aimlOutput = bot.GetAIMLOutputWithLSA(lsaData, aimlKernel, userInput)
            #print("aiml with lsa: ", aimlOutput)
        userEmotion = bot.GetUserEmotion(userInput)
        #print("Emotion of user: ", userEmotion)
        botEmotion = bot.predict_emotion(personalityType, userEmotion)
        #print("Emotion of chatbot: ", botEmotion)
        personalityOutput = bot.GetPersonalityOutput(emotionsOptionData, botEmotion, aimlOutput)
        personalityCategory = "enneagram: " if personalityType.isdigit() else "mbti: "
        return render_template("index.html", personalityCategory = personalityCategory, botOutput1 = personalityOutput, userInput1 = userInput, personalityType=personalityType)

if __name__ == '__main__':
    app.run(debug=False)