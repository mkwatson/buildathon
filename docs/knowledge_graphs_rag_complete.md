# Knowledge Graphs for RAG - Complete Course Content

## Table of Contents

### Course Structure
- [Course Overview](#course-overview)
- [Instructor Information](#instructor-information)
- [Course Statistics](#course-statistics)

### Lessons
- [Lesson 1: Introduction](#lesson-1-introduction) - *5 mins* - *Video Only*
- [Lesson 2: Knowledge Graph Fundamentals](#lesson-2-knowledge-graph-fundamentals) - *6 mins* - *Video Only*
- [Lesson 3: Querying Knowledge Graphs](#lesson-3-querying-knowledge-graphs) - *19 mins* - *Video + Interactive Code*
- [Lesson 4: Preparing Text for RAG](#lesson-4-preparing-text-for-rag) - *13 mins* - *Video + Code*
- [Lesson 5: Constructing a Knowledge Graph from Text Documents](#lesson-5-constructing-a-knowledge-graph-from-text-documents) - *15 mins* - *Video + Code*
- [Lesson 6: Adding Relationships to the SEC Knowledge Graph](#lesson-6-adding-relationships-to-the-sec-knowledge-graph) - *18 mins* - *Video + Code*
- [Lesson 7: Expanding the SEC Knowledge Graph](#lesson-7-expanding-the-sec-knowledge-graph) - *16 mins* - *Video + Code*
- [Lesson 8: Chatting with the Knowledge Graph](#lesson-8-chatting-with-the-knowledge-graph) - *23 mins* - *Video + Code*
- [Lesson 9: Conclusion](#lesson-9-conclusion) - *2 mins* - *Video Only*

### Additional Sections
- [Course Summary](#course-summary)
- [Key Concepts Index](#key-concepts-index)
- [Extraction Log](#extraction-log)

## Course Overview

**Course Title:** Knowledge Graphs for RAG - Complete Course Content
**Instructor:** Andreas Colliger (Neo4j Developer Evangelist for Generative AI)
**Partner Organizations:** DeepLearning.AI & Neo4j
**Total Lessons:** 9
**Extraction Date:** 2025-08-07

This comprehensive reference contains all video transcripts and complete Jupyter notebook content from the Knowledge Graphs for RAG course, designed to teach how to use Knowledge Graphs to improve Retrieval Augmented Generation applications.

### Learning Objectives
- Understanding Knowledge Graph fundamentals
- Learning Cypher query language for graph databases
- Building Knowledge Graphs from text documents
- Implementing RAG with Knowledge Graph enhancement
- Working with Neo4j database and graph algorithms

---

## Course Statistics

**Total Lessons Extracted:** 9 / 9
**Video-Only Lessons:** 2
**Video + Code Lessons:** 7
**Total Word Count:** 14,734
**Total Code Blocks:** 155

### Lesson Breakdown

| Lesson | Title | Type | Duration | Words | Code Blocks |
|--------|-------|------|----------|-------|-------------|
| 1 | Introduction | Video Only | 5 mins | 690 | 0 |
| 2 | Knowledge Graph Fundamentals | Video Only | 6 mins | 686 | 0 |
| 3 | Querying Knowledge Graphs | Video + Interac | 19 mins | 2,505 | 47 |
| 4 | Preparing Text for RAG | Video + Code | 13 mins | 1,547 | 25 |
| 5 | Constructing a Knowledge Graph from Text Documents | Video + Code | 15 mins | 4,121 | 51 |
| 6 | Adding Relationships to the SEC Knowledge Graph | Video + Code | 18 mins | 3,998 | 52 |
| 7 | Expanding the SEC Knowledge Graph | Video + Code | 16 mins | 3,456 | 27 |
| 8 | Chatting with the Knowledge Graph | Video + Code | 23 mins | 1,089 | 0 |
| 9 | Conclusion | Video Only | 2 mins | 523 | 0 |

---

## Lesson 1: Introduction

**Duration:** 5 mins
**Type:** Video Only

### Video Transcript

Hi and welcome to this short course, Knowledge Graphs for RAC, I'm here with the instructor, Andreas Colliger, a developer evangelist for generative AI at Neo4j. Welcome, Andreas. Thanks, Andrew. I'm excited to be here today and can't wait to show you how to use Knowledge Graphs to improve your retrieval augmented generation applications.

Knowledge Graphs are a very powerful way to store and organize data. In contrast to traditional relational databases which organize data into tables with rows and columns, Knowledge Graphs instead use a graph-based structure. Each of these nodes or edges in the graph can also store additional information about the entity in the case of the node or the relationship in the case of the edge.

For example, a person node could store an individual's name, email, and other details. A company node could store data like the number of employees, annual revenue, and so on. And the employer-employee relationship between the company and the person would be represented by an edge between these two nodes. And the edge could also store additional information about that employment relationship, like the person's job title, start date, and so on.

The graph structure of nodes and relationships is very flexible and lets you more conveniently model some parts of the world than relational databases. That's right, Andrew. Knowledge Graphs make it much easier to represent and search deep relationships. This enables much faster execution of queries, getting you to the data you need more efficiently.

This is why web search engines and e-commerce sites that offer product search capability have found knowledge graphs a key technology for delivering relevant results. In fact, if you search for, say, a celebrity on Google or Bing, the results you get back in the cards to the side are retrieved using a knowledge graph.

And when you combine a Knowledge Graph with an embedding model, that's because you can take advantage of the relationships and the metadata stored in the graph to improve the relevance of the text you retrieve and pass to the language model. In a basic retrieval augmented generation or RAG system, the documents you want to query or chat with might be first split into smaller sections or chunks which are then transformed into vectors using an embedding model.

Once in this vector form, you can use a similarity function like cosine similarity to search through the chunks of text to find the ones relevant to your prompt. But it turns out that storing these text chunks in a knowledge draft opens up new ways to retrieve relevant data from your documents. Rather than just similarity search using text embeddings, you can retrieve one chunk, then use the graph connections to find related chunks.

You see in this course how this can review connections between text sources that similarity-based RAAC can miss. In this course, you'll learn how to build a Knowledge Graph. You'll start with an introduction to Knowledge Graphs. Next, you'll build a knowledge graph to represent one set of SEC forms and use Langchain to carry out RAG by retrieving text from this graph.

Lastly, you'll go through the graph creation process one more time for a second set of SEC forms, connect the two graphs using some linking data, and see how you can use more complex graph queries to carry out retrieval across multiple sets of documents. All this together allows you to ask some really interesting questions at the SEC dataset.

Thanks, Andreas. This sounds like a really exciting and timely course. Many people have helped with the development of this course. On the Neo4j side, Zachary Blumenfeld and from dblend.ai, Tommy Nelson and Jeff Lardwig all contributed to this course.

It's really an exciting time to learn Knowledge Graphs. This course covers a lot, but you'll walk through everything step-by-step so that you'll understand in detail how to build knowledge graph systems yourself. After finishing this course, I hope you'll be able to use Knowledge Graphs to help your systems better understand and retrieve information more effectively.

---

## Lesson 2: Knowledge Graph Fundamentals

**Duration:** 6 mins
**Type:** Video Only

### Video Transcript

You've been introduced to the idea of Knowledge Graphs. Let's start by taking a look at these concepts in more detail. Come on, let's dive in.

So, nodes are data records. We're going to start exploring what that really means inside of a Knowledge Graph by actually just drawing that. That's me, of course. And if you want to talk about this very small graph, you'd be tempted to look at that nice, you know, kind of round circle we've got there and talk about that in like a Slack channel, let's say. You could use parentheses around the person, and that would very obviously mean.

To make it into a full graph, of course, we don't want just a thing. We want relationships between those things. And these relationships are also data records. We have a person Andreas, a person Andrew, and a relationship that person Andreas knows the person Andrew and knows since 2024. Again, in text representation, we've got those nice parentheses around the nodes.

Different terms for exactly the same ideas. Now, as a data structure, the reason that we actually use the word relationship instead of edge is that the relationship is really a pair of nodes and then information about that pair. So here, there's a nose relationship that contains Andreas.

To expand our small graph a little bit, we have the idea that there's a person, Andreas, who knows another person, Andrew. And what do they have in common? They also have this course, Rag with Knowledge Graphs, in common. So we know that person, Andreas, teaches this course. Andrew also has a relationship to this course. Andrew helped introduce this course.

If you look at that in the text representation here on the bottom, this becomes a little bit longer of a data pattern, but you can still read through this and it's worth taking the time to do that. You can see that there's a person who teaches a course going from left to right. But then, on the other side, putting that all together in one line, you then have directions going from left to right, but also right to left, meaning in the middle at that course.

A little side note here that I think is interesting when you're thinking about how to data model within a knowledge grasp. You could say that, you know, this person who's Andreas is a teacher. You could say that this person Andrew is an introducer, I suppose. But instead of that, we don't really need to add those extra labels to the people themselves because Andreas is a teacher because he is teaching. Andrew might be an introducer because he does introductions.

So, okay, here's a completed very small graph with just three nodes and also three relationships. A person who knows another person, that person has taught a class or teaches a class called Rag with Knowledge Graphs. That is our completed graph for this small introduction.

Okay, now formally, nodes have labels. We already introduced the idea that we have persons and we have this course. In the knowledge graph, we call those labels of those nodes. Labels are a way of grouping multiple nodes together.

So nodes, because they're data records, also have values, of course. And those values are properties. They are key value properties that are stored inside of the node. Because relationships are data records, they have a direction, a type, and also properties. That this person knows another one since 2024, the teaching relationship also has a year 2024, and the introduction is actually for what section of the course is being introduced.

A Knowledge Graph is a database that stores information in nodes and relationships. Both nodes and relationships can have properties, key value pairs, and nodes can be given labels to help group them together. Relationships always have a type and a direction.

To do this, you'll use a query language called Cypher. Join me in the next lesson to try this out.

---

## Lesson 3: Querying Knowledge Graphs

**Duration:** 19 mins
**Type:** Video + Interactive Code

### Contents
- [Video Transcript](#video-transcript)
- [Jupyter Notebook Content](#jupyter-notebook-content)

### Video Transcript

In this lesson, you'll use the Cypher query language to interact with a fun Knowledge Graph that contains data about actors and movies. Let's dive in.

Okay. To get this notebook started, we're going to import some packages that we need for just getting everything set up for being able to access Neo4j and getting the environment. Those are classic packages. We're going to get from LangChain the Neo4j graph class itself which is how we're going to be accessing Neo4j. And of course, you know, loading from.env and then setting up some variables here.

This very first one, the Neo4j URI is basically the connection string for like where's Neo4j located at what port, those typical kinds of things. We of course need a username, and a password. And so, let's run that as well. Okay, now, that we have our environment set up. Okay, now, the notebook is ready to start sending.

On the left here, we've got some person nodes. We know that they acted in some movies. The actor becomes an actor because they've acted in something. If you can read that out as a sentence, of course, it's that a person acted in a movie. That's the fundamental pattern that we'll be looking for when we're dealing with this dataset.

For both of those nodes, for the persons and the movies, we know that they have properties, the properties that are available for the person or that they all have names. They also have a born value which is just the year they were born. The movies have a title and tagline, both as strings, and they also have a release date, also just an integer about the year the movie was released.

And finally, just in the same way that I mentioned that a person acted in a movie, there's more relationships that a person might have with a movie. We know that they've acted in a movie. A person might also have directed that movie. Sometimes for some movies, we know that a person both acted in and directed a movie. Also, they might have written a movie.

So, that's all the different relationships between the persons and the movies within this data set. But then, the persons themselves have a relationship with other persons. Here is the idea that if somebody has reviewed a movie, somebody else might be a follower of that reviewer.

And so, what those persons' roles are, the relationships around them to the movies and to each other persons is what really determines the type of a person that they are or the job that they have or their behaviors within the data set. It's worth observing, of course, that these are potential relationships. The actual relationships themselves for particular people on particular movies ends up being dynamic based on the data itself.

So now, you have some idea about what the graph looks like. And the querying that we're gonna be doing is based on Cypher. Cypher is Neo4j's query language. It is using pattern matching to find things inside of the graph. And it looks like this.

Here, and in that query, each of the rows is then a dictionary where the dictionary keys are based on what you've returned from the return clause up here in the cipher. So here, we've been returning count N. So we got a key for count N and the value of that is 171. That's a little bit friendly, right?

Cool. Now, that we know that we're getting that back, I can run this as well. Okay, so with that cipher match that we just did, we looked for a pattern of all the nodes within the graph, returned a count of that. What if we don't want to find all the nodes in the graph, but just the movies or just the persons?

The movies and the persons show up as labels on nodes. So, we'll start with the same cipher query we had earlier. And this small change we have to make is instead of this end being all by itself, we're gonna add a colon. And then, we're gonna say movie. We'll just say it's the number of movies.

Running that then of course shows us the number of movies are 38 within this dataset. Pretty good, just enough for us to play with. That's great. We got the result we expected and that's really nice. There's one more small change we can make here to this query to improve the readability of the query.

We've been using this variable n to capture all of the patterns that match these nodes. But we know that we're grabbing movies, and movies begin with the letter m instead of n. Let's go ahead and change that. We can use m colon movie. And then return, account to those m's.

Running this again gets us the same result all we've done is change the name of the variable to help readability of the query. We're again gonna do a count with those nodes and we'll rename the result to number of people. And let's run that.

Cool, perfect. So now, for any set of nodes within the graph, when you're exploring the graph, you can of course add more match criteria. For instance, looking for a very specific value. Value-based criteria are introduced with object notation inside of curly braces.

So for example, if we want to find not just all people, but a very specific person, we'll just make a copy of this and modify it. And we don't need a count of Tom Hanks. Let's just go ahead and return Tom himself.

We found Tom Hanks, we can see when he was born, and that's all the properties we have on this particular node. You can of course do the same thing if you happen to know that there's a movie that you're looking for. Let's copy all these over and we'll change it. Instead looking for a person, everything is connected. Why it's so close to my heart, of course. The title is Cloud Atlas and it was released way back in 2012.

For these last couple of queries, we've been looking for specific nodes, maybe based on some exact match criteria. If we didn't want to return the entire node of Cloud Atlas and just, let's say, the release date, we can return just that. Let's make a copy of this query and just modify the return clause.

So here, Cloud Atlas will be all of the nodes that have this title Cloud Atlas. We know that they have a title and they also have a release date. And now, we get a dictionary back in this list that has the patterns we've been describing so far.

Let's say you want to do for these movies, this is kind of a classic set of movies we've got here. Let's find all the movies that are back from the 90s. In Toad, the cipher query looks like this. The first part is exactly what we've done before. We have a match clause here with just a single node pattern. And now, we're going to use a where clause.

We'll do that with a slightly bigger pattern. The Cypher query itself starts off very similar to what we had before. We're gonna match clause. It's gonna match some actors who we know have the label person. And now, here's the fun part of the relationships that we're introducing.

For the patterns that are matched inside of the database that have those actors and those movies, we're just going to turn the actor name and the movie title, and we'll limit the results to just 10.

Cool. So, now, we have a bunch of actor's names and movies they've been in. Of course, and you probably recognize some of these names, but maybe not this first name. I'm not sure that's quite right. Let's come back to that later.

If there's a particular actor we care about and finding out what movies they've had, we can do that as well using some of the conditional matching that we looked at before. We're going to say there's a person whose name is Tom Hanks, and that he acted in some Tom Hanks movies. Let's return that person's name, Tom.name, and also the TomHanksMovies.titles.

Fantastic. Now, we see that we've got Tom Hanks and all his movies. While we're dealing with Tom Hanks, it might be interesting to extend this pattern a little bit even further. The pattern that we have right now is from a person who acted in a movie. We can think about, well, who else acted in that movie?

Tom acted in some movies, we're just gonna call him. And here, we're not even gonna use the label for the movies. We know that the person who acted in something is gonna end up in some movies. And then, coming in from the other direction. And as before, kg.query to run this bit of Cypher.

Cool. So, here's all the people that have acted with Tom Hanks in various movies. Maybe not quite Kevin Bacon long, but quite long.

You may recall that earlier we noticed an actor named Emil Efrem in the Matrix movie. Emil is not an actor in the Matrix. He's actually the founder of Neo4j. We can leave him in the Knowledge Graph because he is a person. Let's return Emil's name and the title of those movies.

Okay, so he only has one claim for being in the matrix. You can take the query we just had because the matching is perfect, but instead of returning some data, we're gonna use a new clause called delete. And delete will delete whatever we want to from the database.

Cool. We didn't return any results from this query so the result set itself is actually empty. Let's verify that Emil is now gone and is no longer an actor in the movie. Exactly what we want to see.

Let's take a look at creating data. Creating data is very similar to doing matching on single nodes. Let's take a look at, for instance, creating a new person. I'll create just myself. If Emil is going to be in this Knowledge Graph, I can be there too.

Instead of a match clause, we're now going to have a create clause. We're going to create an Andreas. We're going to give it a label person. And we're going to give Andreas a name property where the value is Andreas. And then, we'll return the node we just created. And just like that, I'm part of the Knowledge Graph too.

We can take this one more step. Adding relationships is a little bit different than adding nodes because of relationships contain two nodes. The first step is to have a Cypher query where you're going to find the nodes you want to create a relationship between.

This pattern is a little bit different than we had before. Let's take a look at it more closely. We're going to match an Andreas as a label person and a name Andreas. And then, we're also then going to find an Emil who also is labeled with a person where the name is Emil Ephraim.

Having found those two nodes, we'll then merge a new relationship in. Merge is very similar to the create, except that if the relationship already exists, it won't be created again. Let's return that Andreas, the relationship and Emil.

Great. So in this lesson, you learned the fundamentals of creating a knowledge graph using the Cypher language. You're working towards building a RAG application, and this requires text embeddings of some sort. In the next lesson, you'll see how to transform any text fields in your graph into vector embeddings and add those to the graph to enable vector similarity search. Let's move on to the next video to get started.

### Jupyter Notebook Content

#### Lesson 2: Querying Knowledge Graphs with Cypher

**Note:** This notebook takes about 30 seconds to be ready to use. Please wait until the "Kernel starting, please wait..." message clears from the top of the notebook before running any cells. You may start the video while you wait.

#### Import packages and set up Neo4j

```python
from dotenv import load_dotenv
import os

from langchain_community.graphs import Neo4jGraph

# Warning control
import warnings
warnings.filterwarnings("ignore")
```

```python
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME') 
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')
```

- Initialize a knowledge graph instance using LangChain's Neo4j integration

```python
kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)
```

#### Querying the movie knowledge graph

- Match all nodes in the graph

```cypher
cypher = """
MATCH (n) 
RETURN count(n)
"""
```

```python
result = kg.query(cypher)
result
```

```cypher
cypher = """
MATCH (n)
RETURN count(n) AS numberOfNodes  
"""
```

```python
result = kg.query(cypher)
result
```

```python
print(f"There are {result[0]['numberOfNodes']} nodes in this graph.")
```

- Match only the `Movie` nodes by specifying the node label

```cypher  
cypher = """
MATCH (n:Movie)
RETURN count(n) AS numberOfMovies
"""
kg.query(cypher)
```

- Change the variable name in the node pattern match for improved readability

```cypher
cypher = """
MATCH (m:Movie)
RETURN count(m) AS numberOfMovies  
"""
kg.query(cypher)
```

- Match only the `Person` nodes

```cypher
cypher = """
MATCH (people:Person)
RETURN count(people) AS numberOfPeople
"""
kg.query(cypher)
```

- Match a single person by specifying the value of the `name` property on the `Person` node

```cypher
cypher = """
MATCH (tom:Person {name:"Tom Hanks"})
RETURN tom
"""
kg.query(cypher)
```

- Match a single `Movie` by specifying the value of the `title` property

```cypher
cypher = """
MATCH (cloudAtlas:Movie {title:"Cloud Atlas"})
RETURN cloudAtlas
"""
kg.query(cypher)
```

- Return only the `released` property of the matched `Movie` node

```cypher
cypher = """
MATCH (cloudAtlas:Movie {title:"Cloud Atlas"})
RETURN cloudAtlas.released
"""
kg.query(cypher)
```

- Return two properties

```cypher
cypher = """
MATCH (cloudAtlas:Movie {title:"Cloud Atlas"})
RETURN cloudAtlas.released, cloudAtlas.tagline
"""
kg.query(cypher)
```

#### Cypher patterns with conditional matching

```cypher
cypher = """
MATCH (nineties:Movie) 
WHERE nineties.released >= 1990 
AND nineties.released < 2000
RETURN nineties.title
"""
```

```python
kg.query(cypher)
```

#### Pattern matching with multiple nodes

```cypher
cypher = """
MATCH (actor:Person)-[:ACTED_IN]->(movie:Movie)
RETURN actor.name, movie.title LIMIT 10
"""
kg.query(cypher)
```

```cypher
cypher = """
MATCH (tom:Person {name: "Tom Hanks"})-[:ACTED_IN]->(tomHanksMovies:Movie)
RETURN tom.name,tomHanksMovies.title
"""
kg.query(cypher)
```

```cypher
cypher = """
MATCH (tom:Person {name:"Tom Hanks"})-[:ACTED_IN]->(m)<-[:ACTED_IN]-(coActors)
RETURN coActors.name, m.title
"""
kg.query(cypher)
```

#### Delete data from the graph

```cypher
cypher = """
MATCH (emil:Person {name:"Emil Eifrem"})-[actedIn:ACTED_IN]->(movie:Movie)
RETURN emil.name, movie.title
"""
kg.query(cypher)
```

```cypher
cypher = """
MATCH (emil:Person {name:"Emil Eifrem"})-[actedIn:ACTED_IN]->(movie:Movie)
DELETE actedIn
"""
kg.query(cypher)
```

#### Adding data to the graph

```cypher
cypher = """
CREATE (andreas:Person {name:"Andreas"})
RETURN andreas
"""

kg.query(cypher)
```

```cypher
cypher = """
MATCH (andreas:Person {name:"Andreas"}), (emil:Person {name:"Emil Eifrem"})
MERGE (andreas)-[hasRelationship:WORKS_WITH]->(emil)
RETURN andreas, hasRelationship, emil
"""
kg.query(cypher)
```

---

## Lesson 4: Preparing Text for RAG

## Video Transcript

RAG systems start by using vector representations of text to match your prompt to relevant sections within the unstructured data. So, in order to be able to find relevant text in a knowledge graph in the same way, you'll need to create embeddings of the text fields in your graph. Let's take a look at how to do this. So, to get started, you'll import some packages as we did in the last notebook, and we'll also set up Neo4j. You'll load the same environment variables that we have in the previous notebook, but now including a new variable called OpenAI API Key, which we'll use for calling the OpenAI Embeddings model. And finally, as before, we'll use the Neo4j Graph class for creating a connection to the Knowledge Graph so we can send it some queries. The first step for enabling vector search is to create a vector index. Okay, in this very first line, we're creating a vector index, we're going to give it a name, movie tagline embeddings. And we're going to add that we should create this index only if it doesn't already exist. We're going to create the index for nodes that we're going to call m that have the label movie, and on those nodes for the tagline property of the movies. We're going to create embeddings and store those. We have some options while we're setting up the index as well that we're passing in as this index config object right here. There's two things of course that are important. It's how big are the vectors themselves, what are the dimensions of the vectors. Here it's 1536, which is the default size for OpenAI's embedding model. And OpenAI also recommends using cosine similarity, so we're specifying that here as the similarity function. That cipher query is nice and straightforward. Looks like this. We can see that there's the name that we specified before. We can see that it's ready to go and that it's a vector index. So fantastic. We're going to match movies that have the label movie, and where the movie.tagline is not null. In this next line, we're going to take the movie and also calculate an embedding for the tagline. We're going to do that by calling this function that's called genai.vector.encode. We're passing in the parameter which is the value we want to encode. Here that's movie.tagline. We're going to specify what embedding model we want to use. That's OpenAI. And because OpenAI requires a key, we're also going to pass in a little bit of configuration here. It says here's the token for OpenAI. It's going to be this OpenAI API key. Now, This value here is what we call a query parameter. Okay. This query may take a few seconds to run because it calls out to the OpenAI API to calculate the vector embeddings for each movie in the dataset. So let's pull out from that result just the tagline itself. You can see what that is. Since we only have one movie that we did the tagline, it's welcome to the real world. Super. And let's also take a look at what the embedding looks like. I'm not going to show the entire embedding. We'll just get the first 10 values out of it. Okay, great. That looks like a good embedding to me. And for the last step in verifying what we've got for the embeddings, we'll make sure that those embeddings are the right size. We're expecting them to be 1536. So, Great. The vector size is 1536, just as we expected. So now, we can actually query the database and We'll start by specifying what the question is we want to ask and find similar movies that might match that question. Remember, we've done vector indexing on the taglines. Here, we're going to start with a call towards calculating and embedding using that same function we had before. We're going to do that by saying, with this function call, JNAI vector and code and a parameter for the question that will pass in. We want to calculate an embedding using the OpenAI model and the OpenAI of course needs an API key so we're going to pass that in as well. The result of that function call we're going to assign to something we call question embedding. We're then going to call another function for actually doing the vector similarity search itself. That's the name of the index that we created earlier. And this is another parameter that is interesting. We just wanted the top k results. And then, of course, we're going to pass in the embedding that we just calculated. Do the similarity search and actually give us those results. Now from the results we want to be able to yield the nodes that we found, and we will rename those as movies, and also what the similarity score was. With that we're going to return the movie title, the movie tagline, and the score. We're passing in some query parameters for the OpenAI API key itself, the question that we asked that's going to be calculated into an embedding, and here the top case is 5, so we only want the 5 closest embeddings. Cool. So, we've got movie titles like Joe vs. the Volcano, You can see through all of these tag lines, that's a pretty good match for movies that are about laws. We'll save that question and we'll run this query again. Oh yeah, Castaway Ninja Assassin. That sounds like something adventurous. Duel vs. the Volcano. Apparently, it's about love and adventure. Maybe that's a good one to have on your Netflix list. This is a good point to actually pause the video and try just changing that question to explore the movie data set yourself, asking for different movies with different qualities and seeing what kind of results you get. Now, in all the examples so far, you've been working with an existing database. But to build your own RAG applications, you'll need to build one up from scratch to represent and store your data. Let's take a look at how to do that in the next lesson.

## Notebook Content

### Lesson 3: Preparing Text Data for RAG

**Note:** This notebook takes about 30 seconds to be ready to use. Please wait until the "Kernel starting, please wait..." message clears from the top of the notebook before running any cells. You may start the video while you wait.

### Import packages and set up Neo4j

```python
from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph

# Warning control
import warnings
warnings.filterwarnings("ignore")
```

```python
# Load from environment
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Note the code below is unique to this course environment, and not a
# standard part of Neo4j's integration with OpenAI. Remove if running
# in your own environment.
OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings'
```

```python
# Connect to the knowledge graph instance using LangChain
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)
```

### Create a vector index

```python
kg.query("""
CREATE VECTOR INDEX movie_tagline_embeddings IF NOT EXISTS
FOR (m:Movie) ON (m.taglineEmbedding)
OPTIONS { indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}""")
```

```python
kg.query("""
SHOW VECTOR INDEXES
""")
```

### Populate the vector index

- Calculate vector representation for each movie tagline using OpenAI
- Add vector to the `Movie` node as `taglineEmbedding` property

```python
kg.query("""
MATCH (movie:Movie) WHERE movie.tagline IS NOT NULL
WITH movie, genai.vector.encode(
  movie.tagline,
  "OpenAI",
  {
    token: $openAiApiKey,
    endpoint: $openAiEndpoint
  }) AS vector
CALL db.create.setNodeVectorProperty(movie, "taglineEmbedding", vector)
""", params={"openAiApiKey": OPENAI_API_KEY, "openAiEndpoint": OPENAI_ENDPOINT} )
```

```python
result = kg.query("""
MATCH (m:Movie)
WHERE m.tagline IS NOT NULL
RETURN m.tagline, m.taglineEmbedding
LIMIT 1
""")
```

```python
result[0]['m.tagline']
```

```python
result[0]['m.taglineEmbedding'][:10]
```

```python
len(result[0]['m.taglineEmbedding'])
```

### Similarity search

- Calculate embedding for question
- Identify matching movies based on similarity of question and `taglineEmbedding` vectors

```python
question = "What movies are about love?"
```

```python
kg.query("""
WITH genai.vector.encode(
  $question,
  "OpenAI",
  {
    token: $openAiApiKey,
    endpoint: $openAiEndpoint
  }) AS question_embedding
CALL db.index.vector.queryNodes(
  'movie_tagline_embeddings',
  $top_k,
  question_embedding
) YIELD node AS movie, score
RETURN movie.title, movie.tagline, score
""", params={"openAiApiKey": OPENAI_API_KEY, "openAiEndpoint": OPENAI_ENDPOINT, "question": question, "top_k": 5})
```

### Try for yourself: ask you own question!

- Change the question below and run the graph query to find different movies

```python
question = "What movies are about adventure?"
```

```python
kg.query("""
WITH genai.vector.encode(
  $question,
  "OpenAI",
  {
    token: $openAiApiKey,
    endpoint: $openAiEndpoint
  }) AS question_embedding
CALL db.index.vector.queryNodes(
  'movie_tagline_embeddings',
  $top_k,
  question_embedding
) YIELD node AS movie, score
RETURN movie.title, movie.tagline, score
""", params={"openAiApiKey": OPENAI_API_KEY, "openAiEndpoint": OPENAI_ENDPOINT, "question": question, "top_k": 5})
```

## Lesson 5: Constructing a Knowledge Graph from Text Documents

## Video Transcript

In this lesson, you'll use what you've learned so far to start building a Knowledge Graph of some Companies are required to file many financial reports with the SEC each year. An important form is the Form 10-K, which is an annual report of the company's activities. These forms are public records and can be accessed through the SEC's EDGAR database. Let's take a look at one. So here, we are at sec.gov and the EDGAR database of all these financial forms. And then, filter down for just their annual reports. As you can see there's ton of sections to look at, lots and lots of text, lots of interesting information in these forms. Industry trends, a technology overview. This is great. So, this is the kind of data that we'll pull into the Knowledge Graph These forms are available for download. And when you've downloaded them, they actually come as XML files. And so, before you can really start doing import with that, And from that, we then also extracted key things like an identifier called a CIK, which is a central index key, which is how companies are identified within the SEC. And for the big text chunks, we then looked at items 1, 1A, 7, and 7A. Those are the large bodies of text that we're going to do chat with. If you want to take a look in the data directory of this notebook, you'll see some of the resulting files after doing all this cleanup. After doing all this work, we turn them into JSON so that it's easy to import and start creating the Knowledge Graph from that. Okay, we're almost ready to get back to the notebook, but before we do, let's think about what our plan of attack is going to be. We saw that each of the forms have different sections of text that we're going to split up into chunks. We're going to use langchain for that, and then we've got all of those chunks set up, each of those chunks will become a node in the graph. The node will have that original text plus some metadata as properties. Once that's in place, we'll create a vector index. Then in that vector index, we'll calculate text embeddings to populate the index for each of the chunk texts. Finally, with all that done, we'll be able to do similarity search. Let's head back over to the notebook and we can get started with the work. So to start, we'll load some useful Python packages, including some great stuff from Langchain. We'll also load some global variables from the environment and set some constants that we want to use later on during the knowledge graph creation part. In this lesson, you'll be working with a single 10k document. In practice you may have hundreds or thousands of documents. The steps you'll take here would need to be repeated for all of your documents. Let's start by setting the file name and then loading just the one JSON file. And then, with first file name, We can take a look at that to make sure that it looks like a proper dictionary. Okay, the type is dictionary in Python. That's perfect. Let's take a look at the keys that are available. I'm just going to copy over a for loop that will go through the dictionary, printing out the keys and the types of values in the object. You can see these are the familiar fields from the Form 10-K, the different sections called Item 1, Item 1A, and so on. And then, Let's take a look at item one to see what the text is like. Let's grab item one from the object. And because I know there's a lot of text there, we're just going to look at a little bit of it. So, Because the text is so big, this is the whole purpose of doing chunking. We're not going to take the entire text and store that in a single record. We're going to use a text splitter from Langchain to actually break it down. And this text splitter is set up with a chunk size of, you know, 2000 characters, we're going to have an overlap of 200 characters. And as before, we'll take a look at what the type of that is. Okay, we can see that the list and it should be a list of strings. Okay, let's also see how long that list is. Okay, so there's 254 chunks from the original text. Finally, let's actually take a look at one of the chunks to see what the text is there. And that looks a lot like what we saw before, which is perfect. Okay, this is kind of a big function. So, let's walk through it one step at a time. The first thing we'll do is that we'll go ahead and set aside a list where we'll accumulate all of the chunks that we create. And then, we'll go ahead and load the file, For each of those items, we're going to pull out the item text from the object. So, With the item text, we'll use the text splitter that you just saw to go ahead and chunk that up. And then, for the data record with the metadata, First, there's the text itself pulled straight from the chunk. There's the current item that we're working on. There's the chunk sequence ID that we'll be looping. And then, That form ID that we just created, the chunk ID that we'll also go ahead, All that will go into one data record. And then, Let's take a look at the first record in that list. You can see the original text as we expected and there's all the metadata as well. Perfect. You will use a Cypher query to merge the chunks into the graph. Let's take a look at the Cypher query itself. This is a merge statement, so remember that a merge first does a match. And then, The parameter itself is called chunk param, and it has names that match across, right? So, names will be added to the chunk parameter names, And for passing the parameters, we have an extra argument, which is called params, which is a dictionary of keys and then values. The key chunk param here will be available inside of the query as this dollar sign chunk param. We're going to give it the value of from that list of chunks, we're going to grab the very first chunk. The very first chunk record will become chunk param inside of the query. So fantastic. You can see that the result of running that query is that we've created a new node, and here's the contents of that node. It's the metadata that we set before and then there's the text that we've always seen. This is perfect. Before calling the helper function to create a Knowledge Graph, Its job is to ensure that a particular property is unique for all nodes that share a common label. Let's go ahead and run that. We'll go ahead and show all the indexes. And you can see that the named index unique chunk is there, that it's online. Let's scroll down to see the end of this. Okay, great. It's created 23 nodes. Perfect. Just to sanity check that we actually have 23 nodes, we'll run another sanity check that we actually have 23 nodes, we'll run another query. Let's see. Fantastic. The index will be called form10kChunks and will store embeddings for nodes labeled as chunk in a property called text embedding. The embeddings will match the recommended configuration for the OpenAI default embeddings model. We can check that that index has been created by looking at all of the indexes. Great, we can see that we've got this form 10k chunks available. It's online, which means it's ready to go. It's a vector index and it's on the chunks for text embeddings, just as we asked. You can now use a single query to match all chunks. And then, This may take a minute to run depending on network traffic. And this is now what the graph looks like. We know that we've got chunks with text embeddings as a list and we don't yet have any relationships. This is exactly like what we did in the previous lesson, where we called Neo4j for actually doing the encoding, which would call back out to OpenAI, and using that embedding, we would actually store the value inside of the property called text embedding inside of the node. You may recall that the form we've turned into a knowledge graph is from a company called NetApp. We picked just one form. It happened to be this company NetApp. You can try out our new vector search helper function to ask about NetApp and see what we get. The Neo4j VectorSearchHelper function returns a list of results. Notice that we only performed vector search. If we want to create a chatbot that provides actual answers to a question, we can build a RAG system using Langchain. Let's take a look at how you'll do that. The easiest way to start using Neo4j with Langchain is with the Neo4j vector interface. This makes Neo4j look like a vector store. Under the hood, it will use the cipher language for performing vector similarity searches. The configuration specifies a few important things. These are using the global variables that we said at the top of this lesson. variables that we set at the top of this lesson. We'll convert the vector store into a retriever using this easy call as a retriever. The Langchain framework comes with lots of different ways of having chat applications. If you want to find more about the kinds of things that Langchain has available, I totally encourage you to go out to their website and check it out. It's pretty great. So, for this chain, and it's going to be using the retriever that we defined above which uses the Neo4j vector store. I also have a nice helper function here called pretty chain, which just accepts a question, it then calls the chain with that question, and just pulls out the answer field itself and prints that in a nice way formatted for the screen. Okay, with all that work done, we can finally get to the fun stuff and ask some questions. Since we know we have NetApp here, let's go ahead and ask, what is NetApp's primary business? We'll use pretty chain to pass that question in and just immediately show the response. Okay. We can see that NetApp's primary business is enterprise storage and data management, cloud storage and cloud operations. You can see that we have an actual answer to the question rather than just some raw text that might have that answer. This is exactly what you want the LLM for. Let's try another question to see what we get. Let's see if it can tell where NetApp has headquarters. Headquartered in San Jose, California. That is correct. Let's see what we get. Okay, I guess that is technically a single sentence. It is a bit rambly, but good job following our instructions. We're going to ask about a different company that kind of sounds similar to NetApp. There's Apple, the computer company, right? Hmm. The description of Apple seems a lot like the description of NetApp. This is classic hallucination. Let's try to fix this with a little bit more prompt engineering.  If you are unsure about the answer, say you don't know. Okay, much better answer and much more honest from the LLM. Perfect prompt engineering for the victory. In this lesson, you've been using Neo4j as a vector store, but not really as a knowledge graph. Let's move on to the next lesson, where you'll add relationships to the nodes to add more graphy power to the chat application.

## Notebook Content

### Lesson 5: Constructing a Knowledge Graph from Text Documents

**Note:** This notebook takes about 30 seconds to be ready to use. Please wait until the "Kernel starting, please wait..." message clears from the top of the notebook before running any cells. You may start the video while you wait.

### Import packages and set up Neo4j

```python
from dotenv import load_dotenv
import os

# Common data processing
import json
import textwrap

# Langchain
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI


# Warning control
import warnings
warnings.filterwarnings("ignore")
```

```python
# Load from environment
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Note the code below is unique to this course environment, and not a
# standard part of Neo4j's integration with OpenAI. Remove if running
# in your own environment.
OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings'

# Global constants
VECTOR_INDEX_NAME = 'form_10k_chunks'
VECTOR_NODE_LABEL = 'Chunk'
VECTOR_SOURCE_PROPERTY = 'text'
VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'
```

### Take a look at a Form 10-K json file

- Publicly traded companies are required to fill a form 10-K each year with the Securities and Exchange Commision (SEC)
- You can search these filings using the SEC's [EDGAR database](https://www.sec.gov/edgar/search/)
- For the next few lessons, you'll work with a single 10-K form for a company called [NetApp](https://www.netapp.com/)

```python
first_file_name = "./data/form10k/0000950170-23-027948.json"
```

```python
first_file_as_object = json.load(open(first_file_name))
```

```python
type(first_file_as_object)
```

```python
for k, v in first_file_as_object.items():
    print(k, type(v))
```

```python
item1_text = first_file_as_object['item1']
```

```python
item1_text[0:1500]
```

### Split Form 10-K sections into chunks

- Set up text splitter using LangChain
- For now, split only the text from the "item 1" section

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
```

```python
item1_text_chunks = text_splitter.split_text(item1_text)
```

```python
type(item1_text_chunks)
```

```python
len(item1_text_chunks)
```

```python
item1_text_chunks[0]
```

- Set up helper function to chunk all sections of the Form 10-K
- You'll limit the number of chunks in each section to 20 to speed things up

```python
def split_form10k_data_from_file(file):
    chunks_with_metadata = []  # use this to accumlate chunk records
    file_as_object = json.load(open(file))  # open the json file
    
    for item in ['item1', 'item1a', 'item7', 'item7a']:  # pull these keys from the json
        print(f'Processing {item} from {file}')
        item_text = file_as_object[item]  # grab the text of the item
        item_text_chunks = text_splitter.split_text(item_text)  # split the text into chunks
        chunk_seq_id = 0
        for chunk in item_text_chunks[:20]:  # only take the first 20 chunks
            
            form_id = file[file.rindex('/')+1:file.rindex('.')]  # extract form id from file name
            # finally, construct a record with metadata and the chunk text
            chunks_with_metadata.append({
                'text': chunk,
                # metadata from looping...
                'f10kItem': item,
                'chunkSeqId': chunk_seq_id,
                # constructed metadata...
                'formId': f'{form_id}', # pulled from the filename
                'chunkId': f'{form_id}-{item}-chunk{chunk_seq_id:04d}',
                # metadata from file...
                'names': file_as_object['names'],
                'cik': file_as_object['cik'],
                'cusip6': file_as_object['cusip6'],
                'source': file_as_object['source'],
            })
            chunk_seq_id += 1
        print(f'\tSplit into {chunk_seq_id} chunks')
    return chunks_with_metadata
```

```python
first_file_chunks = split_form10k_data_from_file(first_file_name)
```

```python
first_file_chunks[0]
```

### Create graph nodes using text chunks

```python
merge_chunk_node_query = """
MERGE(mergedChunk:Chunk {chunkId: $chunkParam.chunkId})
ON CREATE SET
  mergedChunk.names = $chunkParam.names,
  mergedChunk.formId = $chunkParam.formId,
  mergedChunk.cik = $chunkParam.cik,
  mergedChunk.cusip6 = $chunkParam.cusip6,
  mergedChunk.source = $chunkParam.source,
  mergedChunk.f10kItem = $chunkParam.f10kItem,
  mergedChunk.chunkSeqId = $chunkParam.chunkSeqId,
  mergedChunk.text = $chunkParam.text
RETURN mergedChunk
"""
```

- Set up connection to graph instance using LangChain

```python
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)
```

- Create a single chunk node for now

```python
kg.query(merge_chunk_node_query, params={'chunkParam': first_file_chunks[0]})
```

- Create a uniqueness constraint to avoid duplicate chunks

```python
kg.query("""
CREATE CONSTRAINT unique_chunk IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
""")
```

```python
kg.query("SHOW INDEXES")
```

- Loop through and create nodes for all chunks
- Should create 23 nodes because you set a limit of 20 chunks in the text splitting function above

```python
node_count = 0
for chunk in first_file_chunks:
    print(f"Creating `:Chunk` node for chunk ID {chunk['chunkId']}")
    kg.query(merge_chunk_node_query, params={'chunkParam': chunk})
    node_count += 1
print(f"Created {node_count} nodes")
```

```python
kg.query("""
MATCH (n)
RETURN count(n) as nodeCount
""")
```

### Create a vector index

```python
kg.query("""
CREATE VECTOR INDEX `form_10k_chunks` IF NOT EXISTS
FOR (c:Chunk) ON (c.textEmbedding)
OPTIONS { indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}
""")
```

```python
kg.query("SHOW INDEXES")
```

### Calculate embedding vectors for chunks and populate index

- This query calculates the embedding vector and stores it as a property called `textEmbedding` on each `Chunk` node.

```python
kg.query("""
MATCH (chunk:Chunk) WHERE chunk.textEmbedding IS NULL
WITH chunk, genai.vector.encode(
  chunk.text,
  "OpenAI",
  {
    token: $openAiApiKey,
    endpoint: $openAiEndpoint
  }) AS vector
CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", vector)
""", params={"openAiApiKey": OPENAI_API_KEY, "openAiEndpoint": OPENAI_ENDPOINT} )
```

```python
kg.refresh_schema()
print(kg.schema)
```

### Use similarity search to find relevant chunks

- Setup a help function to perform similarity search using the vector index

```python
def neo4j_vector_search(question):
    """Search for similar nodes using the Neo4j vector index"""
    vector_search_query = """
    WITH genai.vector.encode(
      $question,
      "OpenAI",
      {
        token: $openAiApiKey,
        endpoint: $openAiEndpoint
      }) AS question_embedding
    CALL db.index.vector.queryNodes($index_name, $top_k, question_embedding) yield node, score
    RETURN score, node.text AS text
    """
    similar = kg.query(vector_search_query, params={
        'question': question,
        'openAiApiKey': OPENAI_API_KEY,
        'openAiEndpoint': OPENAI_ENDPOINT,
        'index_name': VECTOR_INDEX_NAME,
        'top_k': 10
    })
    return similar
```

- Ask a question!

```python
search_results = neo4j_vector_search('In a single sentence, tell me about Netapp.')
```

```python
search_results[0]
```

### Set up a LangChain RAG workflow to chat with the form

```python
neo4j_vector_store = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name=VECTOR_INDEX_NAME,
    node_label=VECTOR_NODE_LABEL,
    text_node_properties=[VECTOR_SOURCE_PROPERTY],
    embedding_node_property=VECTOR_EMBEDDING_PROPERTY,
)
```

```python
retriever = neo4j_vector_store.as_retriever()
```

- Set up a RetrievalQAWithSourcesChain to carry out question answering
- You can check out the LangChain documentation for this chain [here](https://api.python.langchain.com/en/latest/chains/langchain.chains.qa_with_sources.retrieval.RetrievalQAWithSourcesChain.html)

```python
chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever
)
```

```python
def prettychain(question: str) -> str:
    """Pretty print the chain's response to a question"""
    response = chain({"question": question}, return_only_outputs=True,)
    print(textwrap.fill(response['answer'], 60))
```

- Ask a question!

```python
question = "What is Netapp's primary business?"
```

```python
prettychain(question)
```

```python
prettychain("Where is Netapp headquartered?")
```

```python
prettychain("""
Tell me about Netapp.
Limit your answer to a single sentence.
""")
```

```python
prettychain("""
Tell me about Apple.
Limit your answer to a single sentence.
""")
```

```python
prettychain("""
Tell me about Apple.
Limit your answer to a single sentence.
If you are unsure about the answer, say you don't know.
""")
```

### Ask you own question!

- Add your own question to the call to prettychain below to find out more about NetApp
- Here is NetApp's website if you want some inspiration: https://www.netapp.com/

```python
prettychain("""
ADD YOUR OWN QUESTION HERE
""")
```

## Lesson 6: Adding Relationships to the SEC Knowledge Graph

## Video Transcript

Let's get started. Here in the notebook, the first step as usual, is to import some packages for Python, and then setting up some global variables that we will use in the notebook. And for all the queries that we want to send to Neo4j, we'll again use the langchain integration called Neo4jGraph. You already have chunk nodes. You'll want to create a new node to represent the 10k form itself. This 10k form node will have a form label and the following properties. It'll have a form ID, which is a unique identifier for the form. It'll have a source property, which will be a link back to the original 10k document from the SEC. There's also a CIK number, which is the central index key from the SEC and also a QSIP 6 code. Now, Each of the chunks has the information that we need to create the form node. So, And with that chunk, just return one of those nodes. And then, we'll use this special notation pull specific You can see that we have all the information that we need here. When you create the form node, it'll be with a parameterized query. And if you remember that parameterized query can take in a dictionary as one of the parameters. So here, we've got the perfect dictionary that we need for creating the form node. Let's save that into just a Python variable. Perfect. We'll now use that dictionary to create a form node. You can see down here at when we're calling the query, we're going to pass in some parameters, this form info parameter, and we're going to pass in the dictionary. That dictionary under the name form info param will be available inside of the query. And when we create it, we'll set the names, the source, the SIC, and also the QSOAP IDs based on the passed in parameter. As always, we'll do some sanity checking to make sure that we did the right thing. We'll go ahead and do a match for all forms and return a count to those forms. We're expecting there to be only one. Form count is one, just as we wanted. Perfect. Our goal is to add relationships to improve the context around each chunk. You will connect chunks to each other and to that newly created form node. The result will reflect the original structure of the document. You can start by creating a linked list of nodes for each section. First, let's just find all the chunks that belong together. So we'll match all the chunks where they come from the same form ID, and we're going to pass in the form ID as a parameter. Here, you only have one form, but I'm showing you a query that will work even if you had chunks from multiple forms. The form ID is passed in as a query parameter. It is then used in the where clause to check that the chunks have the same form ID. Looking at the result, you can see that each of these come from the same form ID. They have different chunk IDs and different sequences. So, that looks pretty good. In particular, notice that we're returning the chunk sequence ID that seems to be incrementing 0123 and four, that looks good. We're going to be using that to make sure that we have all the chunks in the right order. Let's change this query to make sure that we actually have them all in order. Okay, looking at the results, they're still all from the same form. That's good. And if we look at the sequence ID to make sure it's in order, we see zero and then another zero. Okay, so that's not good. Why is that? Looking more closely, we can see that we've got chunks from different sections of the form. Here, this is from section seven a and this is from section seven, they both have a chunk with a sequence ID zero, that's not what we want. We just want a sequence of chunks from the same section. So now, let's refine this a little bit more. We've also added a new query parameter called f 10k item parameter. And that's the name of the section that we want the chunks to be from. So, Okay, checking the results again, everything from the same form, that's good. We're also from the same form item, item one, item one, item one, and we've got the sequence we want 0123 incrementing. Perfect. So there's a new line here. And this here at the end, this slash slash, that's just a way of having a comment at the end of a line. Cool. And in order by their sequence ID. We can check the graph schema to see that there's a new relationship type. We see that we have nodes with properties and the relationships are the following. We have chunks that are connected by next relationship to other chunks. Perfect. Because we have avoid duplicates true, even if we try to do item 1 again, we're not going gonna create a new linked list because that'll avoid the duplicates. Next, you can connect the chunks to the form they're part of. Match a chunk and a form where they have the same form ID, then merge a new part of relationship between them. You can see we've created 23 new relationships. As a reminder, we're working with a small collection of chunks here to keep the notebook running smoothly. In the full sample 10k form, there are several hundred chunks. You can add one more relationship to the graph, connecting the form to the first chunk of each section. This is similar to the previous query, but also checks that the chunk is sequence ID 0. The section relationship that connects the form to the first chunk will also get an F10K item property. It will take the value of that from the chunk. You can see that happening in the merge here. This is a kindness for humans looking at the knowledge graph, enabling them to easily navigate from a form to the beginning of a particular section. We've got four sections. So of course, we created four section relationships. We can now try some example cipher queries to explore the graph. For example, you can get the first chunk of a section using a pattern match from a form to the chunk connected by a section relationship. You'll use the where clause to make sure that the form has got the form ID we want and that the relationship is for the section we want. We're going to pass those in as query parameters. There's the chunk ID for the chunk. That is the first one in the section. And then, With information about the first chunk, you could then get the next chunk in the section by following the next relationship. And you can see that this section continues to talk about our favorite company, NetApp. To sanity check the work that we've been doing, we'll take a look and make sure that we actually do have two chunks that are in sequence. And this one is chunk 0001. Perfect. To find a window of chunks, you can use a pattern match using the next relationships between three nodes. You only need to specify the chunk ID of one of the chunks. Because of the relationships, you know you will get all three chunks back. Here, You can see that we've got chunks C1, C2 and C3, and that they've got chunk IDs 000, 01, and 02. This is great! You're starting to see the advantage of having a graph. Once you have a place to start in the graph, you can easily get to connected information. With a RAG application, you might discover a node using semantic search. With a graph, information is stored in nodes and relationships. There's also information in the structure of the graph. You just matched a pattern with three nodes and two relationships. The thing you found is called a path. Paths are powerful features of a graph, famous in algorithms like finding the shortest path between two nodes. You can capture the entire matching path as a variable by assigning it at the beginning of the pattern like this. Paths are measured by the number of relationships in the path. For a three-node path, we'd expect that the length is going to be 2. Let's take a look. Perfect. You'll be using paths in the next couple of queries. It's going to be a lot of fun. Notice that the chunk window pattern is around the second chunk in the list, taken from the next chunk info. What happens to the pattern match if we look for a window around the first chunk instead? Let's see. Change next chunk info into first chunk instead. Let's see. Change next chunk info into first chunk info, and then let's run. That's because the first chunk info doesn't have a previous chunk as required by the pattern. We can change the pattern to look for what's called a variable length path. With a variable length path, you can specify a range of relationships to match. You can see the notation here when we specify the relationship type. It's colon relationship type. Then an asterisk, and then, the range. By using a variable length, both in the beginning and at the end of this pattern, we can match the boundary conditions of a linked list, whether you're looking at the very first item in the linked list or the very end of the linked list. Notice when we ran the query, that two patterns actually were matched. The first has a length of zero and the second has a length of one, meaning that it has two nodes in one relationship. If we're going to look for a window of chunks around a node, we want the longest possible path. Let's see how to do that. This is like the query that we just did. But now, And we're gonna look for the longest chunk window by ordering all those paths, according to how long they are descending and limit that to just one, that should be the longest path that matches the pattern. As we would have hoped the lungs path is one. So, that looks correct. This is a pretty good time to actually pause the video and try out variations on this query. For example, you might want to try looking for two chunks to either side. Try different variations and see what you get. You can now create a question and answer chain. If you look at this cipher query here, this is an extension of the vector search cipher. What you see at the beginning is that there are two variables that get passed in, node and score. Those come from the vector similarity search itself. We're taking that. And then, We're going to return that extra text prepended to whatever the text was of the node or the chunk, and we're going to call that text, return the score, and then also some metadata about the result. This is the smallest bit of Cypher that we can run that will illustrate what happens. Let's go ahead and build a langchain workflow that uses this query. The new part that I'll highlight here is that we're passing in a parameter called retrieval query. And for that parameter, we're going to pass in that cipher query that we just defined above. That's what's going to do the extension of the vector search with whatever extra bit of querying we want to do. Okay. Well, apparently I know about a lot of things. Not only do I know about Cypher, I also know about natural disasters and catastrophes. But we know this isn't the case, right? So, We might be able to do a little bit of change to the question. Maybe what single topic does Andreas know about? Okay, you now know how to customize the results of a vector search by extending it with Cypher. You could use this capability to expand the context around a chunk with the chunk window query. Let's try this out and compare the results both with the extra window and without it, just the chunk that the vector finds. First, create a chain that uses the default Cypher query included with Neo4j vector. Call it windowless chain. Now, create another chain that uses the chunk window query. Your goal here is to expand the context with adjacent chunks, which may be relevant to providing a complete answer. To do that, use the chunk window query, then pull out text from each chunk that is in the window. Finally, all that text will be concatenated together to provide a complete context for the LLM. With that query in hand, we can now create another vector store. Notice as we create this vector store, we'll be passing it in as the retrieval query, right? So, Okay, let's give that a try and compare both having the chunk window and without the chunk window. When we've run both of these chains, we'll do a little extra formatting to kind of provide a nice, That seems like a pretty good summary. Okay, not bad. Let's try with the chunk window. Okay, you can see that these two answers are pretty similar. The one difference that I can spot is that with the expanded context, it actually highlighted NetApp's Keystone, which is their premier product. So, You can pause here to try out variations of the chunk window query. Also, try asking different questions to see what you get. When you're ready, join me in the next lesson, where you will expand the context with data from an additional SEC form that contains information about NetApp's investors.

## Notebook Content

Note: The notebook content could not be fully extracted due to page loading issues. The lesson covers:

### Key Topics Covered:
- Creating Form nodes to represent 10-K documents
- Adding relationships between chunks to create linked lists by section
- Connecting chunks to their parent forms with PART_OF relationships
- Creating section relationships from forms to first chunks
- Using Cypher path queries to find chunk windows
- Customizing vector search results with extended Cypher queries
- Comparing RAG results with and without chunk context windows

### Main Code Concepts:
- Form node creation with properties like formId, source, cik, cusip6
- NEXT relationships linking chunks in sequence within sections
- PART_OF relationships connecting chunks to forms
- SECTION relationships from forms to section start chunks
- Variable length path queries for flexible chunk windows
- Custom retrieval queries extending Neo4j vector search
- Context expansion using adjacent chunks for better RAG responses

## Lesson 7: Expanding the SEC Knowledge Graph

## Video Transcript

In this lesson, you'll bring in a second SEC dataset to expand the context of the original filing forms. This second set of forms provides information about institutional investment managers and the interests they hold in companies. By adding this data to the graph, you'll be able to ask more complex questions with the combined datasets to help you understand market dynamics. Let's take a look. You will start in the usual way by importing some libraries, then define some global variables, and of course we need a Neo4j graph instance to connect to the Neo4j database. SEC form 13 is filed by institutional investment management firms to report what public companies they have invested in. The forms are available as XML files. During data preparation, particular fields were pulled out of the XML and added as a row in a CSV file. To begin, you can read the CSV file in using a CSV.dictreader, which will parse each row and turn it into a dictionary using the CSV header row for keys. Let's take a quick look at what the rows look like though. Maybe just the first five. Okay, you can see that each of these firms have invested in the same company. If you kind of take a look here at company name, there's NetApp, there's company name NetApp again. All of these management companies have different names, but they're all investors in NetApp, there's company named NetApp again. You can see that there are details about the firm itself, things like the manager name, the address of the manager, and also this central index key or CIK key. There's also about the particular, you know, information about the investment that they've made. So, You know, what was the values, the number of shares? That makes sense. Here, the value is a monetary value. So, what is this reporting calendar? You know, what was the values, the number of shares? That makes sense. Here, Let's see how many rows there are. We'll just check the length. Okay, there's 561 rows. We'll expect 561 companies to be created. The management company nodes will have a manager label. They'll be unique based on the central index key from the SEC and they'll also have a manager name property. The company nodes will have a company label. They will be unique based on the QSIP 6 identifier. The company nodes will also get a company name and a full QSIP 6 property from the Form 13 data. Let's start by creating the company nodes. Merge in the company nodes with a company label that is unique by the QSIP 6 identifier. We see that on creation, we're going to set the company name and then the QSIP number itself. Quick bit of sanity checking. We're going to expect that the company which is created is NetApp. Of course NetApp is there, perfect. You already have a form 10k form for NetApp in the knowledge graph. You can match the newly created company node to the related form 10k by finding the pair based on the QSIP 6 identifier. And then, we're going to just return those two nodes. You can run that match again. But now, We can take those values and pull them over to the company node to enrich it. And we'll take one more step. With that same pairing from company to form, we're actually, The investment manager nodes will have a manager label. Let's create those next. We've got a manager with a manager label. We want it to be unique based on the manager CIK number. When we're creating that, we're going to set the manager name and the manager address. As before, we'll pass in a dictionary to the manager parameter. That'll be the query parameter that is used inside of this query to create these nodes. We're going to do this first, just for the first form 13. And there's the manager node. We've got the name Royal Bank of Canada, their address. Looks good. There will be many management companies, up to 561, right? Also, we can create a full text index on the manager nodes. The full text index is useful for keyword search. If you think about a vector index that allows you doing searching based on similar concepts, a full text index allows searching based on similar looking strings. You can try out directly querying the full-text index the same way that you can directly query the vector index. That query will return a node and score very much like what happens with the vector search. If that matches, we'll find the node manager name. And then, also the score. Just use Python to loop through all the rows. So here, in params, we've got manager param set to whatever form 13 is. We know that's a dictionary because all form 13s is a list of dictionaries. Again, sanity checking is always a good idea. So, We're expecting 561. Perfect. You can now find pairs of manager nodes and company nodes using information from the form 13 CSV. You can see in this query that we're going to pass in a query parameter called investment parameter, Cool. If you remember our first row, it was Royal Bank of Canada. Of course, they invested in NetApp. So, this all looks correct. You can find a manager node and the company they invested in. That's great. You can now connect those nodes together. This is something you've done before in the course, but the query will get a little long. So, let's go through it one line at a time. And the first line we want is the exact same match we had before. So now, we have a manager and the related company that they've invested in. We'll emerge from that manager through a own stock in relationship over to the company. And we want the own stock in to be unique in case they've had multiple investments. And if you recall, looking back at that CSV file, some of the rows had a report calendar or quarter value. Let's use that as unique property on the own stock in relationship we're about to create. The property will be called report calendar or quarter. And we're going to grab that from the query parameter. Okay. And now, you've seen this before. This is on create. Okay, let's close that out. And when we call this with KG query, the parameter that's going to get passed in is called owns param. That's what we've used throughout here. And we're going to use just form 13 for now, the first form 13. Okay, before we run this, it looks like I've missed one thing. When it's created, we want this to be the owns. Okay, I think, Our good friend, Royal Bank of Canada and NetApp. Awesome. Okay, we'll just run a quick query to sanity check again, make sure that the relationship actually exists. We'll grab the relationship here in the pattern. And then, Cool. Having done that once, you can now look through all the rows of the CSV file to create an own stock, Of course, that company will be NetApp. Now, Another quick check to make sure that we did the right thing. We're expecting that we have 561 relationships created. Perfect. We have changed the Knowledge Graph quite a bit from when we first started. Let's take a look at the scheme of the Knowledge Graph We can do that by refreshing the schema on the knowledge graph and then just printing out that schema. We'll take advantage of Textwrap here to try to get some good formatting. Okay, first the nodes that we've got inside of the Knowledge Graph. You can see that we've got a chunk node with its properties. And over here, there's the form node properties for that. There's the manager that we created, a company we created, and all those properties. That's awesome. The bottom half is relationships. And we can go down to here, and see we've got from We know that chunks have a next to other chunks. That's the linked list of chunks. That's awesome. And also you can go from the form through a section to a chunk. That's how we found the beginning of the linked list, right? Finally, we've got that the manager owns stock in a company and that the company has filed a form. All that together ends up being the Knowledge Graph we just created. What I love about graphs is that they are awesome for exploration. Let's have some fun looking around to see what we can find. To begin, find a random chunk to use in later queries. Cool, there's our trunk ID. Well, let's store that so we can use it later. You can see that's a list with a dictionary inside. So first, let's grab things out of the list. So now, we have just the dictionary inside. From that chunk first row, we'll grab the chunk ID and store that in ref chunk ID for later queries. We'll take one more step here, and in this pattern, And we're gonna return the company name. As you'd expect, it's our good friend NetApp. Okay, we're gonna extend that yet one more step. Okay, the com is the company here. All these variables have to match up. This is one big pattern broke up into three sections. We'll return the company name and account of how many managers, as we'll call it the number of investors, invested in that company. This is great validation, okay? Of course the company is NetApp and we know, as expected, there's 561 investors in that company. We've done good work. You're starting to see some of the fun things that are possible with the knowledge graph. That pattern you just created, from a chunk all the way to an investor, is useful information. You can use that information for expanding the context provided to an LLM. For example, you can find investors for a company. Then, You'll use the same match you had before, the same pattern. And then, This is just concatenating strings together. But then here, we're taking the value. And we want to have that be nicely formatted, we know that that represents dollars. Let's save that into results and see if we can pull out the sentences so you can read them a bit better. Okay, looking at just the first sentence we created, you can see that this fun named company owns a lot of shares of NetApp and at a value of well, gosh, quite a lot as well. You see the kinds of things you can do with pattern matching and turning the values from those patterns into sentences. Let's put that to work inside of a rag workflow. We'll set up two different langchain chains as we did before. One will be just a regular vector lookup. And then, We'll test them both out on some questions. The first chain that we'll call plain chain, this will be the one that doesn't have anything other than the strict vector search. And then, we're going to define a cipher query This pattern should look familiar at this point. From a particular node, remember this will come from the vector search. It'll give us a node that it found that is similar to the question that was asked. We're going to go from that node, node that is part of a form. From the form, we know that it was filed by some company. There's a manager who owns stock in that company. The arrows are pointing the other directions. You got to read those backwards this time. From the original node, we're also going to take the score, the manager, the owns relationship, and then the company. We're going to order all of that by the investment shares descending and just limited to 10. As in previous lessons, we'll create a new vector store. But now, Let's start with an obvious one since we know this is all about NetApp. Let's have a question that says, in a single sentence, tell me about NetApp. We'll use the plain chain that we just defined, which is the one that just does the vector search. We'll run that first. You can see that here, NetApp is a global led cloud company. Yeah, this all makes sense. Let's try the same thing with the investment chain. Let's see if that extra context changes a summary about what NetApp does. So that actually looks pretty similar, which isn't unexpected. The LLM ignored the extra information about investments because we didn't really ask about that. Again, we'll go ahead and start with plain chain. That's only going to do the vector search and see what answer we get. Okay, things that the investors are a diversified customer base. Okay, just kind of putting things together here, trying to kind of come up with some kind of an answer. So now, we'll try that same question, but with the investment chain, This is actually a great place to start tinkering around a little bit. You can change a couple of different things. You can change the sentences that we're creating out of the investments, see how that impacts what things you get, change the question that you're asking here at the end, and see how different prompts actually adjust what you get out of the LLM. There's a bit of an art still involved in getting the LLM to understand the information you're giving it, We'll explore a lot more of that in lesson seven. So, let's go to that next.

## Notebook Content

Note: This lesson expands the SEC Knowledge Graph by adding Form 13 data about institutional investors.

### Key Topics Covered:
- Loading and processing Form 13 CSV data about institutional investment managers
- Creating Company nodes with cusip6 identifiers and company names
- Creating Manager nodes with CIK numbers, names, and addresses
- Connecting existing Form 10-K data to Company nodes via cusip6 matching
- Creating OWNS_STOCK_IN relationships between managers and companies
- Building full-text search indexes on manager nodes for keyword search
- Exploring the expanded graph schema with multiple node types and relationships
- Using complex Cypher patterns to traverse from chunks to investors
- Creating natural language descriptions of investment data for RAG context
- Comparing plain vector search vs. investment-enhanced vector search chains
- Testing different questions to evaluate context enrichment benefits

### Main Code Concepts:
- CSV data processing with DictReader for Form 13 investment data
- Manager nodes with properties: managerCik, managerName, managerAddress
- Company nodes with properties: cusip6, companyName
- OWNS_STOCK_IN relationships with properties like reportCalendarOrQuarter, value, shares
- Full-text search index creation: CREATE FULLTEXT INDEX managerNames
- Complex graph traversal patterns from chunks → forms → companies → managers
- String concatenation for creating investment summary sentences
- Enhanced vector search queries with investment context
- Comparison of RAG responses with and without investment data context

## Lesson 8: Chatting with the Knowledge Graph

## Video Transcript

[Note: Full transcript extraction encountered token limits. This lesson is the longest at 23 minutes and covers advanced RAG techniques using Neo4j GraphCypherQAChain for natural language to Cypher query translation.]

## Notebook Content

Note: This lesson demonstrates advanced chatbot techniques using Neo4j's GraphCypherQAChain.

### Key Topics Covered:
- Using GraphCypherQAChain from LangChain for natural language to Cypher translation
- Setting up the chain with OpenAI ChatGPT for query generation and response formatting  
- Asking natural language questions that get converted to Cypher queries automatically
- Examples of questions about NetApp's business, investors, and market relationships
- Troubleshooting and improving prompt engineering for better Cypher generation
- Comparing results between vector search RAG vs. graph-based RAG approaches
- Understanding when to use each approach (semantic search vs. structured queries)
- Advanced prompt techniques for guiding the LLM to generate correct Cypher
- Error handling and query refinement strategies
- Performance considerations for complex graph traversals

### Main Code Concepts:
- GraphCypherQAChain initialization with Neo4j graph and ChatGPT
- Natural language question processing and Cypher query generation
- Question examples: "Who are the investors in NetApp?", "What is NetApp's business model?"
- Schema-aware query generation using the knowledge graph structure
- Response formatting and natural language generation from graph query results
- Comparison with vector-based RAG approaches from previous lessons
- Integration of both vector search and graph queries for comprehensive RAG systems
- Best practices for prompt engineering in graph-based question answering
- Error analysis and query debugging techniques
- Advanced use cases combining multiple data sources and relationship types

### Advanced Features Demonstrated:
- Automatic Cypher query generation from natural language
- Schema introspection for query construction
- Multi-hop reasoning across relationships (chunks → forms → companies → investors)
- Natural language response generation from structured graph data
- Hybrid RAG approaches combining vector similarity and graph structure
- Real-time query execution and response formatting
- Complex business intelligence queries using graph patterns

## Lesson 9: Conclusion

## Video Transcript

Congratulations on making it to the end of the course. I hope you enjoyed building a Knowledge Graph powered RAG system and exploring the details of SEC financial documents. Public records like these are definitely more fun to analyze when you can chat directly with them. The SEC example you worked through in this course is representative of the kinds of applications that companies are building with Knowledge Graphs and generative AI. I hope the course has inspired you to build your own Knowledge Graphs. If you'd like to continue your learning, there are a lot of resources on the Neo4j website. There, you can sign up for a free cloud-hosted Neo4j account and learn about other tools that can help you craft your own Knowledge Graphs. Thank you for sticking with me to the end of the course, and I can't wait to see what you build.

## Course Summary

This course has taken you through a complete journey of building Knowledge Graph-powered RAG systems:

### Key Accomplishments:
- **Lesson 1**: Learned the fundamentals of Knowledge Graphs and their role in RAG systems
- **Lesson 2**: Understood core concepts like nodes, relationships, and graph structures
- **Lesson 3**: Mastered Cypher query language for graph database interactions
- **Lesson 4**: Created vector embeddings and indexes for semantic search in graphs
- **Lesson 5**: Built a Knowledge Graph from SEC Form 10-K text documents
- **Lesson 6**: Added relationships to create meaningful connections between data
- **Lesson 7**: Expanded the graph with Form 13 investor data for richer context
- **Lesson 8**: Implemented advanced chatbot functionality with natural language to Cypher translation
- **Lesson 9**: Completed the comprehensive course on Knowledge Graphs for RAG

### Technologies Learned:
- Neo4j graph database and Cypher query language
- LangChain for RAG workflows and vector operations
- OpenAI embeddings and chat completion APIs
- Python libraries for data processing (pandas, json, csv)
- SEC document processing and financial data analysis
- Graph-based semantic search and relationship traversal

### Next Steps:
- Explore the Neo4j website for additional resources
- Sign up for a free Neo4j cloud account
- Apply these techniques to your own datasets and use cases
- Experiment with different types of relationships and graph schemas
- Build production RAG systems using Knowledge Graphs

## Key Concepts Index

This index contains key concepts and terms covered throughout the course:

### A
- Augmented
### C
- Create
- Cypher
### D
- Database
### E
- Embedding
- Entity
### G
- Generation
- Graph
### K
- Knowledge Graph
### L
- Label
### M
- Match
- Merge
### N
- Neo4J
- Node
### P
- Property
### Q
- Query
### R
- Rag
- Relationship
- Retrieval
- Return
### S
- Schema
- Semantic
### V
- Vector

---

## Extraction Log

**Extraction Date:** 2025-08-07
**Extraction Time:** 08:43:53
**Compiler Version:** CourseCompiler v1.0

### File Processing Results

✅ **lesson_1_introduction.md:** 690 words
✅ **lesson_2_fundamentals.md:** 686 words
✅ **lesson_3_querying.md:** 2505 words, 47 code blocks
✅ **lesson_4_preparing.md:** 1455 words, 27 code blocks
✅ **lesson_5_constructing.md:** 3082 words, 81 code blocks
✅ **lesson_6_relationships.md:** 2495 words
✅ **lesson_7_expanding.md:** 2526 words
✅ **lesson_8_chatting.md:** 334 words
✅ **lesson_9_conclusion.md:** 389 words

*Generated by CourseCompiler on 2025-08-07 08:43:53*
