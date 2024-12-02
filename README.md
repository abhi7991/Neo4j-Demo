<div style="text-align:center;">
    <img src="logo.png" width=200>
</div>

# CineSphere
Movie recommendations using Knowledge Graphs and LLMs

> [Technical Presentation](CineSphere%20-%20Presentation.pptx)<br>
> [Technical Documentation](other%20files/CineSphere%20-%20Technical%20Documentation.pdf)<br>
> [About Me - Abhishek](Abhishek%20Shankar%20-%20About%20Me.pptx)<br>

## Introduction 


CineSphere is your ultimate destination for discovering world-class entertainment. Our platform not only brings you the best of global cinema but also answers all your questions about your favorite movies, actors, and celebrities. Some of the features available on the 
product are. 

- Interactive User Interface and experience
- Personalised movie recommendations based on user preferences
- Q&A bot to cater to user queries
- Immediate movie suggestions tailored to user moods
- Wide variety of world cinema

## Architecture Diagram
<img src="CineSphere_Flow.png" alt="Product Flow">


## Implementation

1. Recommendation - CineSphere uses the Node Similarity which is available through Neo4j's Graph Data Science plugin. Users will be allowed to signup and be produced a set of movies. Upon selecting the movies the graph will capture all relationships and start running a similarity algorithm. The more the user interacts with the recommendation
the better results they will be able to get. 

2. Similarity - The number of questions which might pop up while watching a movie like information regarding the cast or having a debate about an actors movie. Well we got you covered. The product leverages the similarity between the various movie nodes and the interaction of the genres, users, actors and crew members around the movie nodes. The app does this on the fly and can be accessed through the chat bot. For this functionality CineSphere uses Neo4j's graph data science.

3. Q&A - The application also facilitates questions ontop of the dataset where in it will provide information about the movies like "Give me all the top grossing movies of an actor" or "How much a particular movie made in the box office?". This is done with the help of LangChain and OpenAI to launch a GraphQAChain. 

All the 3 main features are compiled to form a multi-agent tool. This multi-agent tool helps route the questions to the appropriate function and helps handle a variety of questions.
  

**Steps to Reproduce**

1) Clone the repo 
2) Create an .env file with the following variables - 

```
NEO4J_URI = ''
NEO4J_USER = ''
NEO4J_PASSWORD = ''
OPENAI_API_KEY=''

```