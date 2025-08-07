# Buildathon Graph & AI Tech Textbook

## Introduction

Welcome to the **Graph & AI Buildathon** reference textbook. This guide covers 15 key technologies spanning **graph databases, LLM tooling, and rapid demo stacks**. Each section provides a **cheat sheet** of commands and syntax, **example use cases with code**, **setup instructions**, and **common pitfalls with debugging tips**. We also include a **comparison matrix of agent frameworks**, guidance on **tool selection trade-offs**, and tips for **deploying polished demos quickly**. Use the interactive table of contents below to navigate:

* **LLM APIs (OpenAI, Anthropic, Vectara)** – How to call large language model APIs in Python/TypeScript, key parameters, and differences.
* **Neo4j + Cypher** – Graph database queries with Cypher, CRUD operations on nodes/relationships, and data modeling patterns.
* **LangGraph (LangChain)** – Building complex multi-agent workflows with LangChain’s LangGraph library.
* **CrewAI** – Creating collaborative AI agent “crews” with specialized roles, flows vs. crews, CLI setup, and performance notes.
* **Microsoft Autogen** – Multi-agent conversation framework for LLMs, using Assistant and UserProxy agents to let LLMs collaborate.
* **OpenAI Agents SDK** – Lightweight SDK to build agentic apps with minimal abstractions, using agents, tools, handoffs, and tracing.
* **LangSmith & Helicone (LLM Observability)** – Tools for logging, monitoring, and debugging LLM calls, integrating into apps for live telemetry.
* **Vercel AI SDK + Next.js** – Rapid front-end integration of AI with React/Next.js, using hooks like `useChat` and streaming UI updates.
* **Replit Deployments** – One-click cloud deployment of apps from Replit, deployment types (Autoscale, Static, etc.) and how to monitor them.
* **Supabase Vector (pgvector)** – Using Postgres as a vector database (pgvector extension) for semantic search, creating indexes and querying with `<->` distance.
* **Cloudflare Workers AI + Vectorize** – Running AI at the edge with Cloudflare’s Workers AI, storing embeddings in Vectorize (distributed vector DB).
* **MongoDB Atlas Vector Search** – Adding vector search to MongoDB Atlas, creating a Search index and querying with `$search` and k-NN operator.
* **Streamlit** – Quickly building and sharing data apps in Python with a web UI, key Streamlit widgets and deployment to Streamlit Cloud.
* **Python + TypeScript Rapid Stack** – Strategies to combine Python (for ML/backend) and TypeScript (for frontend/serverless) for fast prototyping and iteration.
* **Prompt Engineering & RAG** – Best practices for LLM prompt design (role prompts, examples, CoT) and Retrieval-Augmented Generation patterns for grounded outputs.

Armed with this textbook, you’ll be ready to **build and demo AI applications** quickly and confidently. Let’s dive in!

---

## 1. LLM APIs (OpenAI, Anthropic, Vectara)

Large Language Model APIs allow you to leverage powerful models via cloud services. This section covers using **OpenAI**, **Anthropic (Claude)**, and **Vectara** APIs, including authentication, key endpoints, and usage patterns.

**Cheat Sheet – Key Commands & Concepts:**

* **Authentication:** Acquire API keys from each provider’s dashboard (e.g. OpenAI key from account page, Anthropic key from console) and set them as environment variables or in code. For OpenAI Python, set `openai.api_key` or use `OPENAI_API_KEY` env. For Anthropic’s SDK, use `ANTHROPIC_API_KEY` env or pass key to client constructor.
* **Endpoints/SDKs:**

  * *OpenAI:* Use the `openai` Python/Node library. Key methods: `openai.Completion.create()` for completion models and `openai.ChatCompletion.create()` for chat models. The chat API expects a list of messages with roles (“system”, “user”, “assistant”).
  * *Anthropic:* Use the `anthropic` SDK (Python or TypeScript). The main method is `client = anthropic.Anthropic()` then `client.messages.create(...)` providing the model (e.g. `"claude-2"`), prompt, etc. Anthropic’s API uses a similar message format (system and user prompts).
  * *Vectara:* Vectara is an LLM-powered semantic search platform; you typically index data then query via their API. It supports **hybrid search** (keyword + vector) and can call integrated LLMs for answers. Use their REST API or SDK to **index documents** and then issue a **query** request with natural language. (Authentication uses OAuth2 client credentials).
* **Model IDs:** OpenAI model names include `"gpt-3.5-turbo"`, `"gpt-4"` (and snapshot versions like `-0613`), or embedding models like `"text-embedding-ada-002"`. Anthropic models are named like `"claude-2"`, `"claude-instant-1"`, etc. Vectara provides certain built-in models or allows connecting external ones.
* **Parameters:** Temperature (controls randomness), max\_tokens (response length), top\_p (nucleus sampling) are common to OpenAI/Anthropic. OpenAI also has `frequency_penalty` and `presence_penalty` to reduce repetition. Anthropic uses a simpler set (temperature, max\_tokens) and has a concept of “Claude’s tone” via system prompts.
* **Rate Limits:** Note each API’s rate limits. OpenAI has TPM (tokens per minute) and RPM (requests per min) quotas by model. Anthropic and Vectara also impose limits. Exceeding these causes errors – implement exponential backoff or request rate increase for production.

**Example Use Cases & Code Snippets:**

* *OpenAI ChatCompletion (Python):* Use a list of messages for a conversation. For example, to get a joke from GPT-3.5:

  ```python
  import openai
  openai.api_key = "<YOUR_OPENAI_KEY>"
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a friendly assistant."},
        {"role": "user", "content": "Tell a joke about statistics."}
      ],
      temperature=0.7
  )
  print(response["choices"][0]["message"]["content"])
  ```

  This will yield an assistant reply (in this case, a joke). The `messages` structure allows the system role to set behavior and the user message to pose the query. Adjust `temperature` or use `max_tokens` to limit length.

* *Anthropic Claude (Python SDK):*

  ```python
  import anthropic
  client = anthropic.Anthropic()  # API key auto-picked from env
  prompt = "You are a world-class poet. Respond with a short poem about the sea."
  # Claude uses a simple API: send system prompt via 'system' param and user prompt in messages
  resp = client.messages.create(model="claude-2", 
                                system="You are a world-class poet.",
                                messages=[{"role":"user","content":[{"type":"text","text":"Why is the ocean salty?"}]}],
                                max_tokens=100)
  print(resp.content)
  ```

  This sends a user question to Claude with a system instruction to respond as a poem. The result might be a poetic explanation of ocean salinity. Anthropic’s API returns the completion in `resp.content`.

* *Vectara Semantic Search (HTTP API):* Pseudo-code:

  1. **Indexing**: Send documents (text with optional metadata) via `POST /v1/index` to your corpus.
  2. **Query**: `POST /v1/query` with JSON like `{"query": "How to optimize my database?", "top_k": 5}`. Vectara will return relevant passages (and can generate answer if configured).
     Example using curl:

  ```bash
  curl -X POST "https://api.vectara.com/v1/query" \
       -H "Authorization: Bearer $VECTARA_TOKEN" \
       -d '{ "query": {"text": "How to optimize Neo4j performance?"}, "top_k": 3 }'
  ```

  The response includes the top matches with scores and text. Vectara’s platform can also directly integrate with LLMs to **reduce hallucinations** and provide cited answers.

**Setup Instructions (Minimal Friction):**

* **OpenAI:** Install the official library (`pip install openai` or `npm install openai`). Set your API key in an env var or directly in code. No additional config needed. Ensure your network calls to `api.openai.com` are allowed.
* **Anthropic:** Install their SDK (`pip install anthropic` or `npm install @anthropic-ai/sdk`). For Python, ensure `ANTHROPIC_API_KEY` is in your environment. If not using the SDK, you can call the HTTP endpoint directly with an `Authorization: Bearer <key>` header (the API URL is `https://api.anthropic.com/v1/complete` for older v1 completion API, or newer multi-message endpoints as per docs).
* **Vectara:** Sign up for an account to get credentials (Customer ID, Corpus ID, API Key or OAuth token). Use their Python client or REST API. You may need to enable the **LLM Assistant** mode if you want it to produce answers. Vectara requires creating a corpus and indexing data before querying. Refer to Vectara’s docs for quickstart.

**Common Pitfalls & Debugging:**

* *Token Limits:* Each model has context length limits (e.g. GPT-3.5 \~4K tokens, GPT-4 8K or 32K, Claude 100K for Claude-2). If you get truncated responses or errors like “context length exceeded,” trim your prompts or switch to a model with larger context.
* *Rate Limit Errors:* If you see HTTP 429 or errors about rate limit, implement retry with backoff. For OpenAI, check response headers for `Retry-After`. For Anthropic, ensure you’re not sending more than allowed TPS. Consider enabling **batch** calls or fine-tuning queries to reduce usage.
* *API Errors & Logging:* Enable verbose logging during development. For OpenAI Python, you can set `openai.logging = true` to see request/response details (or log the `response.usage` object which contains token counts and request IDs). For Anthropic, handle exceptions from the SDK (they often include an error message). Using an observability tool like **Helicone or LangSmith** (see Section 7) can automatically log all LLM API calls along with latency and errors for you.
* *Model-Specific Quirks:* OpenAI’s `ChatCompletion` might return a finish\_reason of `"length"` if it hit max\_tokens – in that case, you may need to request more tokens or adjust the prompt to encourage completion. Claude might refuse certain requests with a safety warning (as Anthropic has built-in safety). In such cases, adjust the phrasing or explicitly allow it via system prompt if appropriate (Anthropic supports an optional `--force` parameter in their playground to override harmless refusals). Vectara’s answers might be brief by default; you can prompt it to be more verbose or retrieve more passages (`top_k`).

<br>

## 2. Neo4j + Cypher

**Neo4j** is a leading graph database, and **Cypher** is its query language for pattern matching in graphs. In a Buildathon, Neo4j can power features like knowledge graphs, network analysis, or recommendation systems. This section provides a Cypher cheatsheet and examples to get you productive with graph data.

**Cheat Sheet – Cypher Commands & Syntax:**

* **Creating Nodes:** Use `CREATE` with labels and properties. Example: `CREATE (p:Person {name: 'Alice', age: 30})` creates a node with label Person and properties name, age. Multiple nodes/relationships can be created in one statement separated by commas.
* **Creating Relationships:** To relate nodes, specify a pattern with relationship arrows. Example: `MATCH (a:Person {name:'Alice'}), (b:Company {name:'Neo4j'}) CREATE (a)-[:WORKS_AT]->(b)` creates a WORKS\_AT relationship from Alice to Neo4j. Use square brackets `[:TYPE]` to name the relationship and optionally add `{...}` for relationship properties.
* **Matching Patterns:** `MATCH (n:Label)-[r:REL_TYPE]->(m:OtherLabel) WHERE n.property = value RETURN n, r, m` is the common form. The `MATCH` clause uses parentheses for nodes and `-[]-` for relationships. Omitting a direction (`--`) matches relationships in either direction. Use labels to narrow matches (or leave label off to match any node). The `WHERE` clause adds value filters (like `n.age > 25`). Example:

  ```cypher
  MATCH (p:Person)-[r:FRIENDS_WITH]-(p2:Person)
  WHERE p.name = 'Alice' AND p2.age < 30
  RETURN p2.name AS youngFriend
  ```

  This finds names of Alice’s friends under 30. Boolean operators `AND`, `OR`, comparison `<, >, =, <>` are supported.
* **Returning Data:** The `RETURN` clause specifies what to output. Use aliases with `AS` to rename columns (e.g., `RETURN p.name AS Name, count(*) AS Num`). Use aggregation functions like `count(n)`, `avg(n.score)`, etc., along with `GROUP BY`-like behavior via aggregation without listing non-aggregated fields.
* **Updating Data:**

  * *SET:* Add or update properties. `MATCH (p:Person {name:'Alice'}) SET p.age = 31, p.status = 'Active'` modifies existing properties or adds new ones.
  * *DELETE:* Remove nodes or relationships. Must detach relationships first if deleting a node (or use `DETACH DELETE n` to remove a node and its relationships in one go). Example: `MATCH (p:Person {name:'Bob'}) DETACH DELETE p`.
  * *MERGE:* Find or create patterns atomically. `MERGE` tries to match given pattern; if not found, it creates it. Use this to avoid duplicates (like ensuring only one node per unique key). E.g., `MERGE (c:City {name:'London'}) ON CREATE SET c.created = timestamp()`.
* **Indexes & Constraints:** Creating an index on a label improves lookup speed: `CREATE INDEX FOR (p:Person) ON (p.email)` (Cypher in Neo4j 4+ syntax) will index Person.email. Unique constraints ensure no two nodes share a property: `CREATE CONSTRAINT ON (p:Person) ASSERT p.id IS UNIQUE`. Use `SHOW INDEXES` or Neo4j Browser UI to verify indexes. In Memgraph (a Neo4j fork), syntax is `CREATE INDEX ON :Person(name)`, and checking via `SHOW INDEX INFO;`.

**Example Use Cases & Cypher Queries:**

* *Social Network Query:* Find friends-of-friends (2nd degree connections).

  ```cypher
  MATCH (me:Person {name: "Alice"})-[:FRIENDS_WITH]->(friend)-[:FRIENDS_WITH]->(fof)
  WHERE fof <> me   // ensure the FoF is not Alice herself
  RETURN DISTINCT fof.name AS friendOfFriend;
  ```

  This matches a path Alice -> friend -> fof, and returns the second-hop names. `DISTINCT` is used to avoid repeats if multiple paths lead to the same person. The `<>` operator checks inequality (here ensuring we don’t return Alice).

* *Graph Analytics:* Find the shortest path between two nodes. For instance, if we have a graph of cities connected by roads:

  ```cypher
  MATCH p = shortestPath( (c1:City {name:"London"})-[:ROAD_TO*..10]-(c2:City {name:"Paris"}) )
  RETURN p, length(p) AS hops;
  ```

  This finds a path up to 10 hops (edges) long between London and Paris using breadth-first search internally. The `shortestPath()` function ensures the returned path is minimal. The `*..10` in the relationship means traverse between 0 and 10 edges.

* *Recommendation Query:* “Friends who like the same movie as me.”

  ```cypher
  MATCH (me:User {name:"Alice"})-[:LIKES]->(m:Movie)<-[:LIKES]-(other:User)
  WHERE other <> me
  RETURN other.name AS recommended_friend, collect(m.title) AS sharedInterests;
  ```

  This finds users (`other`) who liked any movie that Alice (`me`) liked. The `collect(m.title)` aggregates the common movies into a list for context. This could be used to recommend Alice to connect with those users.

**Setup Instructions:**

* **Neo4j Desktop or AuraDB:** For local dev, Neo4j Desktop provides a DB you can run queries on with a GUI. For cloud, Neo4j AuraDB Free Tier can be spun up quickly. Once running, you can connect via Neo4j Browser (web UI) or use drivers.
* **Language Drivers:** Install the Neo4j driver for your language. E.g., for Python: `pip install neo4j` (then use the `GraphDatabase` driver class to run Cypher from code). For JavaScript/TypeScript: use the official neo4j-javascript driver (`npm install neo4j-driver`). Example (Python):

  ```python
  from neo4j import GraphDatabase
  uri = "neo4j+s://<your-instance>.databases.neo4j.io"  # Aura or bolt://localhost
  driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))
  with driver.session() as session:
      result = session.run("MATCH (n) RETURN count(n) AS nodeCount;")
      print(result.single()["nodeCount"])
  ```

  This prints the count of all nodes. Replace the URI and credentials as appropriate (for local Neo4j, uri might be `bolt://localhost:7687`).
* **Sample Data:** Importing data is straightforward via Cypher `CREATE` statements or by CSV. Neo4j Browser’s “GUIDES” can import sample datasets (e.g., Movie DB). For quick practice, use the built-in Movie Graph: open Neo4j Browser and run `:play movie-graph` to load it.

**Common Pitfalls & Debugging:**

* *Missing Index causing slowness:* If queries with `WHERE` on properties are slow, ensure you created an index or constraint for that property. Use `EXPLAIN` or `PROFILE <query>` in Neo4j Browser to see the query plan – if you see “Node By Label Scan” instead of “Node Index Seek”, you are missing an index for that lookup.
* *Variables vs Labels:* In Cypher, `()` around a pattern uses variables for nodes. For example, `MATCH (Person {name:'Bob'})` is actually interpreting `Person` as a label (shorthand without the `:` is deprecated). Proper syntax: `MATCH (p:Person {name:'Bob'})`. A common mistake is forgetting the `:` before label names.
* *Updating large datasets:* Using naive `MATCH` + `SET` on many nodes can be slow. Consider using **apoc** procedures or batching updates (e.g., `USING PERIODIC COMMIT` for large CSV loads). If operations fail mid-way, Neo4j transactions roll back entirely – so break large jobs into smaller transactions if needed.
* *Memory and Depth:* Be careful with queries like `(a)-[*]->(b)` without a length bound – it can attempt to traverse an extremely large subgraph. Always specify a max hops or add filtering conditions, or you risk long execution times or out-of-memory errors. Use Neo4j’s configurable `dbms.query_memory_limit` or streaming results for very large result sets.
* *Concurrent writes:* If multiple processes/threads write to the graph, you may encounter transient `DeadlockDetected` errors (if two transactions try to update the same node). Handle these by retrying the transaction (the Neo4j drivers often have built-in retry logic for deadlocks).
* *Cypher syntax errors:* Check for balanced parentheses and curly braces. For instance, forgetting to close a pattern with `)` or a property map with `}` will throw an error indicating the place it was expected. The error messages are usually informative about the token it didn’t expect. Use the Neo4j Browser to iteratively develop queries (it has autocompletion and highlights).

<br>

## 3. LangGraph (LangChain)

**LangGraph** is a library in the LangChain ecosystem for orchestrating complex workflows of LLM “agents” using a graph (DAG or cyclic graph) structure. It enables multi-step, multi-agent interactions beyond simple linear chains. With LangGraph, you can define nodes (tasks) and edges (transitions) to create loops, conditional branches, and even concurrent agents.

**Cheat Sheet – Key Concepts & Syntax:**

* **Installation:** `pip install langgraph langchain` (LangGraph is an extension of LangChain). Also install any provider-specific packages (e.g. `langchain-openai` for OpenAI LLMs).
* **Graph Structure:** You construct a graph of **Nodes** (each node can run a function, tool, or sub-chain) connected by **Edges** (which determine the execution flow). LangGraph allows cycles – meaning an agent can loop back based on conditions (unlike vanilla LangChain’s acyclic chains).
* **Stateful Execution:** LangGraph maintains a **shared state** that nodes can read/write, enabling memory across steps. The state can hold variables like the conversation history, intermediate results, etc.
* **Agents and Tools:** You can still use LangChain agents within LangGraph nodes. For example, a node could be an “agent node” that decides an action using an LLM, then subsequent nodes perform those actions. LangGraph provides a *chat agent executor* that represents the agent’s state as a list of messages for chat models. Tools can be bound to LLMs via `llm.bind_tools(tools)` (as shown below) so the agent can invoke them.
* **Primitives:**

  * `@tool` – a decorator to turn a Python function into a LangChain Tool (usable by agents). For instance, defining `get_weather(query: str) -> dict` and marking it with `@tool` allows the agent to call `get_weather` when needed.
  * **Node Definition:** In code, nodes can be functions or LangChain Chains. You might explicitly construct a `Graph` by adding nodes and connecting them. Alternatively, LangGraph offers higher-level interfaces to define an agent with certain tools and let it internally form the graph (e.g., `create_react_agent` for a standard Reason+Act loop).
  * **Edge Conditions:** Edges can have conditions to create branching. For example, after a node computes something, an edge can route to different next-nodes depending on the state (if a value exceeds threshold, go to node A else node B). This is done by examining state in-between nodes or by the agent’s decision.

**Example Workflow & Code:**

Let’s outline a simple LangGraph application where an agent can either answer from memory or use tools (web search, weather API) when needed:

1. **Tool Setup:** Define tools for web search and weather queries, then bind to the LLM. For example:

   ```python
   from langchain_community.tools.tavily_search import TavilySearchResults
   from langchain_core.tools import tool
   from langchain_openai import ChatOpenAI

   @tool
   def search_web(query: str) -> list:
       """Search the web for a query."""
       results = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=2).invoke(query)
       return results

   @tool
   def get_weather(city: str) -> str:
       """Get current weather for a city."""
       url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
       data = requests.get(url).json()
       return data["current"]["condition"]["text"] if "current" in data else "Weather not found."

   llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
   tools = [search_web, get_weather]
   llm_with_tools = llm.bind_tools(tools)  # Now the LLM can call tools by name
   ```

   Here we use an OpenAI chat model and give it two tools. The `@tool` functions include docstrings which serve as tool descriptions for the agent. After binding, `llm_with_tools.invoke(prompt)` can execute tool calls as instructed by the prompt.

2. **Simple Graph Agent:** Using **LangGraph’s pre-built REACT agent**, which is a Reason+Act loop agent, we can spin up an agent easily:

   ```python
   from langgraph.prebuilt import create_react_agent

   system_prompt = """Act as a helpful assistant.
   - get_weather: use for questions about weather in a location.
   - search_web: use for general information or current events.
   Only use tools if necessary, otherwise answer directly.
   """
   agent = create_react_agent(model=llm_with_tools, tools=tools, state_modifier=system_prompt)

   # Query the agent:
   result = agent.run("What's the weather in Paris today?") 
   print(result)
   ```

   Under the hood, this agent will decide, based on the prompt, to call the `get_weather` tool for Paris. The result (e.g., "It's rainy in Paris.") is returned. The `create_react_agent` sets up a LangGraph with a loop where the LLM reasons and either produces an answer or a tool action. The `state_modifier` (system prompt) guides its decision when to use each tool.

3. **Custom Graph Definition:** For more complex flows, you might manually build a `Graph` of nodes. For example, say we want a graph with two stages: (A) Agent tries to answer question, (B) If answer is not confident or needs info, use a tool, then loop back. Pseudocode:

   ```python
   from langgraph import Graph, Node

   # Node definitions
   answer_node = Node(lambda state: llm(state["question"]))  # ask LLM directly
   tool_node = Node(lambda state: llm_with_tools(state["question"]))  # allow tools

   # Graph construction
   graph = Graph()
   graph.add_node("try_answer", answer_node)
   graph.add_node("try_tool", tool_node)
   # Add edge: if answer_node's output contains a flag like "I don't know", go to tool_node
   def needs_tool(state):
       return "I don't know" in state.get("answer", "")
   graph.add_edge("try_answer", "try_tool", condition=needs_tool)
   graph.add_edge("try_tool", "try_answer")  # loop back after tool use

   result = graph.run({"question": "Who won the World Cup in 2018?"})
   print(result["answer"])
   ```

   In this conceptual example, the agent first attempts to answer from its knowledge. If the answer indicates uncertainty, the graph condition triggers the tool usage node, which can search the web and update state with info, then returns to try answering again. LangGraph’s **state** would carry over the intermediate info (like search results) so that on the second attempt the agent has more context.

**Setup Instructions:**

* **API Keys & Environment:** Ensure any API keys for tools (like the WeatherAPI, search API, OpenAI API) are set in your environment. LangGraph will use the same keys as LangChain (e.g., `OPENAI_API_KEY`).
* **Project Structure:** No special project structure is needed. You can integrate LangGraph into a Streamlit app or a FastAPI server. Just make sure to preserve the graph or agent object if you want it to have long-term memory (for example, store it in a session state or in memory so it doesn’t reinitialize each time).
* **Resource Requirements:** Running multiple LLM calls in a graph can be slow or expensive. For experimentation, use smaller models or set temperature=0 for deterministic behavior during testing. You might also use LangChain’s support for **local models** (like via `openai_api_base` to point to a local LLM server) if needed to avoid cost.

**Common Pitfalls & Debugging:**

* *Complexity:* It’s easy to design a graph that loops infinitely or becomes too complex. Use **logs** and **visualization**. LangGraph has tracing integration (works with LangSmith) to visualize the agent’s chain of thought. You can instrument the graph or use `LangchainDebug` to log each step. If stuck in a loop, implement a counter in state and break after N iterations for safety.
* *State Management:* Since state is shared, be careful to name state keys uniquely to avoid collisions. For example, if two nodes output to `state["result"]`, one might override the other. You can namespace or structure the state (e.g., `state["step1"]["result"]`). Always check what your nodes return – by default a LangChain LLM call might return a complex object; you may want to extract `text` out of it before storing in state.
* *Error Handling:* If a node fails (throws an exception), the whole graph run may fail unless caught. Wrap tool calls with try/except if they can raise (like network errors). You can set up a node to catch errors and decide an alternate path (like if web search fails, maybe try a different source). LangGraph doesn’t automatically retry nodes, so implement retries on API calls within the node logic.
* *Performance:* Multi-agent workflows can be slow if they call many LLM steps. Profile the number of calls – e.g., an agent that overthinks might call the same tool repeatedly. You can mitigate this by adding a cost in the prompt (like “Each tool call has a cost, so try to be efficient”) or by limiting loop iterations. Also consider running certain nodes in parallel if they don’t depend on each other – LangGraph allows concurrency since it’s not strictly sequential (though you’d need to handle merging their results).
* *Compatibility:* Ensure your LangChain version matches the LangGraph version. LangGraph evolves with LangChain; if you see errors importing classes (like `langchain_core` vs newer `langchain` module naming), check the LangGraph docs for the correct versions. The example above uses `langchain_core.tools` which is valid for LangChain 0.\*, but in LangChain 1.x, tools might be imported differently. If you run into import errors, upgrade/downgrade accordingly or consult the LangChain changelog.

<br>

## 4. CrewAI

**CrewAI** is a framework for building *teams of AI agents (“crews”)* that collaborate to solve tasks. Unlike single-agent frameworks, CrewAI emphasizes multiple agents with specialized roles working together (e.g. a “Researcher” agent, a “Writer” agent, etc.) and provides an orchestration layer called **Flows** for structured processes. It’s designed for autonomy *plus* control, allowing hierarchical agent management and robust production deployments.

**Cheat Sheet – Key Concepts & Workflow:**

* **Installation:** `pip install crewai` (and optional extras: `pip install "crewai[tools]"` to get built-in toolset). CrewAI requires Python 3.10+ and uses an internal tool **UV** for environment management (but for basic use, pip works). After install, run `crewai --help` to see CLI options.

* **Crews vs Flows:**

  * A **Crew** is a group of agents with roles that work **autonomously** towards a goal. For example, one agent might generate a plan, another executes steps. CrewAI Crews allow internal delegation and multi-step reasoning without explicit scripting every step.
  * A **Flow** is a deterministic, step-by-step workflow (like a directed process) that can incorporate agents where needed. Think of Flows as the glue for business logic and conditional sequences, and Crews as the AI decision-makers embedded within.

* **Defining Agents (Crew Members):** You typically define agent roles via YAML or Python classes. Each agent has a **role description**, a **goal**, and optionally a **toolset**. For example, in `agents.yaml`:

  ```yaml
  agents:
    - name: "Analyst"
      role: "Data Analyst"
      goal: "Analyze market trends from data and provide insights."
      tools: ["python", "websearch"]
    - name: "Strategist"
      role: "Strategy Expert"
      goal: "Use insights to develop a market strategy."
      tools: []
  ```

  This might define two agents. CrewAI then can instantiate these as separate LLM-powered entities that can message each other. (In code, you can also do `Agent(role="Data Analyst", goal="...")` directly).

* **Tasks and Process:** You also define **Tasks** which are units of work possibly assigned to agents. A simple Task might be “Analyze dataset X and output key metrics” assigned to the Analyst agent. Tasks can be sequential or parallel and can feed into each other. In YAML (or Python), you list tasks with descriptions and expected outputs. CrewAI uses this to manage the process flow.

* **Execution:** Use `Crew` and `Process` classes to kick off the workflows. Pseudocode in Python:

  ```python
  from crewai import Agent, Task, Crew, Process

  analyst = Agent(role="Data Analyst", goal="Analyze market data for trends")
  strategist = Agent(role="Strategy Expert", goal="Create strategy from analysis")
  analysis_task = Task(description="Analyze Q3 market data", agent=analyst,
                       expected_output="Summary of trends")
  strategy_task = Task(description="Draft market strategy based on analysis",
                       agent=strategist, expected_output="Strategy document")
  # Create Crew and define process (sequential here)
  my_crew = Crew(agents=[analyst, strategist], tasks=[analysis_task, strategy_task],
                 process=Process.sequential)
  result = my_crew.kickoff(inputs={"market_data": data})  # starts the crew
  print(result)
  ```

  This spawns the two agents. The Analyst will perform the first task (likely by LLM prompt using its role and the market data), produce an output which CrewAI stores in state, then the Strategist agent takes that output to produce a strategy. The `kickoff` method runs the crew’s tasks in order (or concurrently if `Process.parallel`).

* **CLI Usage:** CrewAI provides a CLI to scaffold projects. Running `crewai create crew my_project` creates a folder with `crew.py`, `agents.yaml`, `tasks.yaml` etc., pre-populated for you. You can edit those YAMLs and then run `crewai run` to execute. This is helpful in hackathons to get started without writing all boilerplate code.

**Example Scenario & Code:**

*Scenario:* Build a “Content Creation Crew” where one agent generates an outline and another writes the article. We’ll illustrate using CrewAI in code for brevity, though YAML config is an option.

```python
from crewai import Agent, Task, Crew, Process

# Define two agents with distinct roles
outline_agent = Agent(role="Outline Specialist", 
                      goal="Outline the key points for the article based on the topic.")
writer_agent = Agent(role="Writer",
                     goal="Expand an outline into a full blog article with engaging content.")

# Define tasks for each agent
topic = "The Benefits of Graph Databases"
task1 = Task(description=f"Create a detailed outline for an article on '{topic}'",
             expected_output="Bullet-point outline covering main points",
             agent=outline_agent)
task2 = Task(description="Write a 1000-word article following the given outline",
             expected_output="Draft article text",
             agent=writer_agent)

# Link tasks: after outline is done, pass to writer
content_crew = Crew(agents=[outline_agent, writer_agent],
                    tasks=[task1, task2],
                    process=Process.sequential)  # sequential flow
result = content_crew.kickoff(inputs={})  # kickoff triggers execution
print(result["outputs"])
```

Here, when `kickoff` runs: the Outline agent will receive its task prompt (which includes the description and presumably any inputs, here just the hardcoded topic) and produce an outline. CrewAI captures that output and passes it as input to the next task (the Writer agent) automatically. The Writer agent then composes the article. Finally, `result["outputs"]` contains the outputs of all tasks (we can access the article text from there).

*What’s happening internally?* CrewAI’s runtime uses the agent definitions to prompt an LLM behind the scenes. For instance, Outline agent’s prompt might look like: “You are an Outline Specialist. Your goal: Outline the key points for the article. Task: Create a detailed outline on 'The Benefits of Graph Databases'.” The LLM (which CrewAI selects or you configure with a model) returns an outline. CrewAI then takes that text, and in the Writer’s prompt includes it (likely as context, e.g. “You are a Writer... Here is an outline: <outline>. Now write a 1000-word article following it.”). The ability to split roles like this helps ensure coherent multi-step outputs.

**Setup Instructions:**

* **Model Configuration:** By default, CrewAI will use OpenAI API if available (make sure `OPENAI_API_KEY` is set). You can configure custom LLM providers via the `agents.yaml` or in code. For example, to use a local model you could set an environment variable or use CrewAI’s integration with **LiteLLM** or **Ollama** for on-prem models. The docs page “Connect CrewAI to LLMs” covers using Together API, Ollama, etc., if you need alternatives.
* **Running & Deploying:** Develop locally using the CLI or Python API. CrewAI can be integrated into a web app: for instance, you could have a FastAPI endpoint that triggers `crew.kickoff()` and returns the result. Keep in mind kickoff is synchronous; for longer runs you might want async (CrewAI supports `crew.kickoff_async()` to not block). For deployment, you can host it like any Python application. If using the CLI, `crewai pack` might bundle the project.
* **Observability:** CrewAI comes with integration to tracking tools (like LangSmith, Arize Phoenix, etc.) to log agent conversations. Enabling these can help see what each agent said. Also, CrewAI prints logs to console by default with each step’s result if `verbose=True` on the Crew or tasks.

**Common Pitfalls & Debugging:**

* *Synchronization issues:* If using **parallel processes**, be careful that agents don’t conflict. For example, two agents writing to the same file or variable could race. CrewAI’s memory for each agent is isolated, but if they converge on a task, coordinate via the Flow (use `Process.sequential` if order matters).
* *Agent going off-track:* Sometimes an agent might not produce the expected output (e.g., Outline agent writes a full article instead of an outline). To prevent this, carefully craft the **role and goal prompts**. CrewAI uses the `goal` and `task description` to form the prompt, so make sure they are explicit. If needed, add additional instructions or few-shot examples in the `context` field of an agent. You can test agents individually by invoking them outside the crew (CrewAI might allow calling `agent.run(some_input)` if needed, or simulate via an isolated prompt to see how it behaves).
* *Long outputs and truncation:* The Writer agent producing 1000 words might hit model token limits. If using GPT-3.5 (4k context) it should be okay for \~1000 words, but GPT-4 8k is safer. If you get incomplete output, consider chunking the task or explicitly instruct the agent to continue (CrewAI might not automatically handle “continue” signals from the model). You could break the writing task into multiple smaller tasks (intro, section1, etc.) assigned to the same agent, orchestrated by a Flow.
* *Performance and Costs:* A Crew with many agents will multiply your API calls. If each agent uses GPT-4, it could be expensive. Monitor usage: CrewAI can output token usage per task (especially if integrated with observability tools). If cost is an issue, use cheaper models for some roles (maybe GPT-3.5 for draft and GPT-4 for final refinement, etc.). CrewAI allows using different models for each agent if configured (the `connectors` or LLM config per agent can be customized).
* *Troubleshooting Errors:* If `crewai run` fails or hangs, use `--verbose` or check the logs in `logs/` (CrewAI may output logs to a directory). Common errors include missing API keys (it will complain if OpenAI key not set), or version mismatches (e.g., using an older tasks/agents format with a newer version – consult CrewAI’s changelog if that happens). Also ensure your Python environment meets dependencies; some CrewAI tools require extra installs (like `crewai[tools]` for web scraping tools, or a Rust compiler for certain dependencies as noted in install troubleshooting).

**Comparison Note:** CrewAI is often compared to other agent frameworks. CrewAI’s strength is *structured multi-agent collaboration* with high performance (its creators claim 5.7× faster than LangChain’s LangGraph in some tasks). It provides a lot out-of-the-box (100+ built-in tools, etc.). In the next section, we’ll see a matrix comparing CrewAI with LangGraph, Autogen, and OpenAI’s SDK.

<br>

## 5. Microsoft AutoGen

**Microsoft AutoGen** is an open-source framework from Microsoft for creating applications with multiple LLM agents that converse with each other. It provides abstractions for **Assistant agents** (AI assistants that can execute code or tools) and **UserProxy agents** (that can simulate a user or wait for human input) to enable complex interactions. AutoGen’s focus is facilitating cooperation among agents via a **conversation loop**.

**Cheat Sheet – Key Features:**

* **Installation:** `pip install autogen-agentchat` (for Python; requires Python 3.10+). After installing, import from `autogen`. Also ensure you have openai API key or appropriate model endpoints configured, since AutoGen by default uses OpenAI (but it can integrate other models as well).
* **Agent Types:**

  * `AssistantAgent` – an AI agent (usually backed by an LLM like GPT-4) that can respond autonomously without human input. It can be set to write Python code in its responses which AutoGen can execute (for tool use). You typically give it a system message like “You are the coding assistant” and it will attempt tasks.
  * `UserProxyAgent` – an agent that stands in for a human. It can either wait for actual user input or automatically respond using an LLM or code execution if configured. For instance, a UserProxyAgent might represent a software environment that runs code the Assistant writes.
* **Agent Conversation Setup:** To create agents:

  ```python
  from autogen import AssistantAgent, UserProxyAgent
  from autogen.coding import DockerCommandLineCodeExecutor

  assistant = AssistantAgent(name="assistant", llm_config={"model": "gpt-4"})
  exec = DockerCommandLineCodeExecutor()  # environment to execute code
  user_proxy = UserProxyAgent(name="user_proxy", 
                               code_execution_config={"executor": exec})
  ```

  Here we make an Assistant agent using GPT-4 and a UserProxy that will execute code in a Docker sandbox. The `code_execution_config` means whenever the user\_proxy sees a message from assistant containing a code block, it will run it and return the output. This is powerful for tool use: the assistant can generate Python code to call an API, and the user\_proxy executes it, returning results, enabling the assistant to see outcomes and refine code.
* **Conversation Loop:** Start the conversation using `assistant.start_conversation(user_proxy, initial_message="...")` or by sending messages explicitly. AutoGen automates the turn-taking: the AssistantAgent and UserProxyAgent will keep exchanging messages until a termination condition (like a max turns or a specific message) is reached. This is different from manually prompting; AutoGen’s loop lets agents figure out when to stop (or you define a stop criteria).
* **Tools and Functions:** AutoGen doesn’t have a separate “tool registry” like LangChain; instead, it uses the code execution mechanism as a general tool use. By writing code, the agent can do almost anything (call APIs, perform calculations, etc.). You can also integrate **OpenAI function-calling** or other tool APIs by customizing the agent’s behavior (AutoGen’s docs mention it’s extensible, but the main path is through code execution).

**Example – Multi-Agent Conversation:**

Suppose we want an AI pair to solve a coding problem: one agent is the “Coder” and the other is a “Reviewer” checking the code. We use AutoGen to have them discuss and refine a solution.

```python
from autogen import AssistantAgent, UserProxyAgent

coder = AssistantAgent(name="coder", system_message="You are a Python coding expert who writes correct code.")
reviewer = AssistantAgent(name="reviewer", system_message="You are a code reviewer who spots bugs and issues.")
# We can treat one agent as "user" to the other by controlling message flow ourselves:
conversation = coder.initiate_conversation(partners=[reviewer])

# Provide initial user problem for them to solve:
conversation.send_message("We need a Python function to check if a number is prime.")

# Let them converse for, say, 5 back-and-forth turns
for _ in range(5):
    conversation.step()  # processes next agent's turn
    last_msg = conversation.get_last_message()
    print(f"{last_msg['sender']}: {last_msg['content']}")
    if conversation.check_end_state():
        break
```

In this snippet, we manually drive the conversation (alternatively, `AutoGen` might have a higher-level `run_chat()` that handles stepping). The coder agent will likely propose some code for is-prime, the reviewer will critique it, and they iterate. Each `step()` call triggers the next agent to respond based on the latest message. We print each message for visibility.

This example shows how AutoGen uses names and roles, and you can orchestrate multi-turn dialogues. In a real scenario, you might not need `for` loop manual stepping – you could use `conversation.run_until_complete()` or similar (depending on API) to let them talk until a solution is reached.

**Setup Instructions:**

* **API Keys/Models:** AutoGen defaults to OpenAI. Set `OPENAI_API_KEY`. If using Azure OpenAI or others, adjust `llm_config` (AutoGen’s `llm_config` can accept an endpoint or Azure creds). If you want non-OpenAI models, AutoGen supports plugging in huggingface or others via “Using Non-OpenAI Models” guide – this may require additional libs or config. Ensure Docker is installed if using DockerCommandLineCodeExecutor for code tools.
* **Verifying Installation:** Try a basic example from the docs (like a two-agent chat or a self-chat). If you encounter model-related errors (e.g., `401` unauthorized), double-check keys. If code execution in Docker fails, ensure Docker is running and your user has permission. There’s also an option to not use Docker (like `PythonREPLCodeExecutor` that runs code in the local environment – safer for small tasks).
* **AutoGen Studio:** Microsoft provides an “AutoGen Studio” as a UI for designing these agents (likely separate install or VSCode extension). It’s optional but could help visualize conversations.

**Common Pitfalls & Debugging:**

* *Long or Endless Conversations:* Agents might get stuck in a loop (e.g., continuously apologizing or correcting trivial things). You should set a max turn limit or a convergence condition. AutoGen’s `check_end_state()` can detect if a solution is found (if you define what message means “done”). Otherwise, enforce a cutoff: e.g., after N turns, stop and take the coder’s latest code as final.
* *Ensuring Focus:* Provide clear system instructions. Without proper role definitions, the agents might converge to agreeing with each other without accomplishing the goal. In the example, giving one agent the explicit job of finding flaws makes the dynamic more productive. You can also introduce a UserProxyAgent if you want to incorporate actual code execution in the loop – for example, the reviewer could actually run the coder’s code on test cases using a UserProxyAgent with code execution instead of just reading it. That significantly enhances debugging (the reviewer agent can then see actual errors or outputs).
* *Resource Usage:* Each agent turn is an API call. A lengthy discussion can use many tokens (especially if they quote the code back and forth). Mitigate by instructing them not to repeat the entire code every turn, or use shorter messages. You might also use cheaper models (AutoGen can use GPT-3.5 for intermediate turns, and only final check with GPT-4 if needed).
* *Error in Code Execution:* If using code tools via the UserProxy, ensure the environment is safe. The assistant might write infinite loops or destructive code. The Docker executor is sandboxed, but still be cautious – don’t mount host volumes or anything sensitive. If the assistant’s code errors out, the UserProxy will send back the error output as a message to the assistant. Be prepared for the assistant to sometimes not know how to handle raw tracebacks – you may need to include in the system prompt guidance like “If you see an error, fix the code accordingly.”
* *State & Memory:* AutoGen by default doesn’t maintain long-term memory beyond the current conversation (other than what’s in the messages). If your agents need knowledge from earlier chats or an external knowledge base, you have to feed that in. You could implement a retrieval step by injecting relevant info into the system/user message at each iteration (this is outside AutoGen’s core but can be layered on).
* *Comparison to others:* AutoGen excels at scenarios where two or more agents *with distinct abilities* (like coding vs verifying) collaborate spontaneously. However, as noted in the CrewAI section, AutoGen lacks a built-in concept of a directed process – the conversation can wander. For tasks requiring strict sequential steps or heavy integration with non-LLM logic, you might need to wrap AutoGen in a traditional program or combine it with something like Flows (CrewAI) or Orchestration code.

<br>

## 6. OpenAI Agents SDK

The **OpenAI Agents SDK** is OpenAI’s own framework for building agentic AI applications, introduced to simplify multi-step tool-using agents. It provides a minimal set of primitives – **Agent**, **Tool**, **Runner**, **Handoff**, **Guardrails**, etc. – designed to be **lightweight but production-ready**. Essentially, it wraps the function-calling and conversation loop logic in a convenient package.

**Cheat Sheet – Essentials:**

* **Installation:** `pip install openai-agents`. Also requires `openai` library and an OpenAI API key set (OPENAI\_API\_KEY). The SDK is language-agnostic to models (you can use it with OpenAI’s or via LiteLLM to other models), but default is OpenAI chat models.
* **Agent Definition:** Create an `Agent` by specifying a `name` and `instructions` (system prompt). Example:

  ```python
  from agents import Agent
  assistant = Agent(name="Assistant", instructions="You are a helpful coding assistant.")
  ```

  This agent now knows its role. You can also attach tools and guardrails when constructing or later. Tools are just Python functions you add via a decorator or registration (the SDK auto-generates JSON schema for them so the LLM can call them). For instance:

  ```python
  from agents import tool

  @tool
  def add(a: int, b: int) -> int:
      """Adds two numbers"""
      return a + b

  assistant.tools = [add]  # now the agent can use the 'add' tool
  ```

  The SDK automatically formats this so that the agent can call `add` with arguments during conversation.
* **Running an Agent:** Use `Runner` to execute the agent loop.

  ```python
  from agents import Runner
  result = Runner.run_sync(assistant, "Calculate 2+2 and tell a joke about it.")
  print(result.final_output)
  ```

  This will start a session with the assistant agent. Under the hood it will feed the user prompt to the model along with the agent’s instructions, and if the model responds with a function call (like `add` with arguments 2 and 2), the SDK will execute that tool function, feed the result back, and continue until the agent returns a final answer. The `final_output` contains the assistant’s last message. There’s also `Runner.run()` for async usage.
* **Multi-Agent Orchestration (Handoffs):** The OpenAI Agents SDK has a concept of **handoffs**, where one agent can delegate to another. For example, you can set up a triage agent that forwards a query to either a Spanish agent or English agent based on language:

  ```python
  spanish_agent = Agent(name="SpanishAgent", instructions="Respond only in Spanish.")
  english_agent = Agent(name="EnglishAgent", instructions="Respond only in English.")
  triage_agent = Agent(name="Triage", 
      instructions="Decide who should handle the request based on language.",
      handoffs=[spanish_agent, english_agent])
  ```

  The triage agent can then `handoff` the conversation to one of those in its list. This is set up by including special content in its prompt or by the `Runner` supporting multiple agents. In the SDK example, they show how a triage agent uses `handoffs` list to automatically forward the message to the right agent. Essentially, a handoff is treated as a specialized tool call that passes control.
* **Guardrails:** The SDK allows adding guardrails – validation functions that run on inputs or outputs. For instance, you can define a guardrail to ensure an email format is correct before agent uses it. This is advanced usage but worth noting: guardrails can prevent the agent from doing certain things (like if output is disallowed by a regex, etc.).
* **Sessions/Memory:** The Agents SDK automatically keeps conversation history if you use a `Session`. By default, `Runner.run_sync` starts a fresh session. If you want memory across multiple interactions, use `Runner.run_sync(session=my_session, ...)` where `my_session = Runner.start_session(agent)`. This session will carry the conversation.

**Example – Using OpenAI Agents SDK:**

*Hello World:*

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a poetic assistant.")
result = Runner.run_sync(agent, "Describe the sun in one sentence.")
print(result.final_output)
```

This simple usage yields a poetic one-liner about the sun. It’s analogous to a single prompt completion, but internally it’s an agent loop (with no tools or handoffs in this case).

*Agent with Tool Example:*

```python
from agents import Agent, Runner, tool

@tool
def get_length(s: str) -> int:
    """Returns the length of the input string."""
    return len(s)

agent = Agent(name="Assistant", instructions="You can compute string lengths using the get_length tool when needed.")
agent.tools = [get_length]

query = "What is the length of the string 'hello world', and can you reverse it?"
result = Runner.run_sync(agent, query)
print(result.final_output)
```

In this scenario, the agent might parse the query and think: first, I need the length of "hello world". It will call `get_length("hello world")` via the tool interface (the SDK handles the function call format). It gets `11` as result, then it might use that to formulate an answer. Since reversing the string might not be an actual tool, the agent’s LLM can just do it mentally or say it cannot (depending on instructions). But likely, it will output: *“The string 'hello world' has length 11, and reversed it is 'dlrow olleh'.”* The tool was used for the length calculation. This shows how the OpenAI Agents SDK leverages OpenAI’s function calling under the hood (the `@tool` decorated function defines a schema, and the model can produce a function call which the SDK executes, similar to how OpenAI function calling works natively).

**Setup Instructions:**

* Make sure you have **OpenAI API access**. The Agents SDK is best with GPT-4 or 3.5 (with function calling). If you have plugin or function-calling access, it should work out-of-the-box. Otherwise, ensure to use models that support function calls (e.g., `"gpt-3.5-turbo-0613"` or `"gpt-4-0613"` which are function-enabled).
* Adjust settings via environment or `Agent.model_settings`. You can specify temperature or other OpenAI parameters in the agent or session. For example, `agent.model = "gpt-4-0613"; agent.model_config={"temperature":0.2}` if needed – refer to SDK docs for exact syntax or use `openai` env vars.
* **Temporal Integration:** The Agents SDK can integrate with **Temporal.io** for durable agents (to survive restarts), but for a hackathon usage you likely won’t need that. Just be aware it exists for production scaling.

**Common Pitfalls & Debugging:**

* *Function Schema Incompatibility:* The SDK tries to infer the JSON schema from Python type hints for tools. If your function has complex types or no type hints, it might fail. Always include simple JSON-serializable types in tool signatures (int, float, str, bool, dict of such, list of such). If a tool fails schema validation, the agent might not call it. You can see tool schema by printing `tool.schema` if needed.
* *Agent Doesn’t Use Tool:* Sometimes the LLM might ignore the tool and try to solve itself. The instruction needs to nudge it: e.g. “You have a tool get\_length at your disposal. Use it for any length calculations.” Also, ensure the query actually triggers function-calling behavior (OpenAI’s function calling will only call if it judges it’s needed). For testing, you can force a tool call by deliberately asking something that requires it. If absolutely needed, you could wrap the question in a format that suggests function (though generally the model is quite good at using provided functions).
* *Chaining multiple tools:* If the agent needs to use multiple tools sequentially (e.g., tool A gives input to tool B), the current OpenAI function calling mechanism typically handles one function call per turn. The SDK’s `handoffs` concept can cover multi-agent delegation, but for a single agent using multiple tools, it will call one tool, get result, then you call `Runner.run_sync` again or use the continuous session approach to allow it to call the next tool. In short, iterative calls may be needed if your agent doesn’t do everything in one go. Alternatively, you might create a wrapper tool that internally uses others (but that defeats some purpose).
* *Guardrails & Validation:* If you add guardrails (like `@guard` decorators on functions to validate inputs/outputs), be mindful that they stop the agent if conditions fail. This is good for safety (e.g., check if output contains a banned word, then refuse). But during a Buildathon, you might skip heavy guardrails to avoid confusion. If something seems not coming through, check that a guard isn’t triggering.
* *Observability:* The SDK has built-in tracing support – it can log spans of each tool call, etc., which can be visualized if you integrate with LangSmith or print to console. If the agent gets stuck, run with debug logging enabled (perhaps the SDK has an env var like `OPENAI_AGENT_LOGLEVEL=DEBUG` or similar – if not, just print the intermediate `result.intermediate_steps` if available). The `result` object contains `result.log` or similar with full conversation transcript and tool calls; examining that can tell you why an agent might have stopped or given a certain answer.

<br>

## 7. LangSmith & Helicone (LLM Observability)

Building with LLMs and agents requires good **observability** – logging prompts, responses, usage, errors, etc. **LangSmith** (by LangChain) and **Helicone** are two tools to help monitor and debug your AI apps. LangSmith offers tracing, evaluation, and dataset management tightly integrated with LangChain. Helicone is an open-source proxy that logs all your LLM requests (across providers) with metrics like latency and cost.

**Cheat Sheet – Logging Tools:**

* **LangSmith:**

  * *Setup:* Install with `pip install langsmith`. Sign up on langsmith.langchain.com to get an API key. Set `LANGSMITH_API_KEY` env and enable tracing by `LANGSMITH_TRACING=true`.
  * *Usage:* You can use LangSmith without LangChain integration by manually wrapping calls, but easiest is if you’re already using LangChain – it can auto-log chains, agents, and even tool usage. For example, in code:

    ```python
    from langsmith import traceable
    from langsmith.wrappers import wrap_openai

    openai_llm = wrap_openai(OpenAI(temperature=0))  # wraps an OpenAI LLM to auto-log
    @traceable
    def rag_answer(question):
        # your retrieval-augmented generation logic
        docs = vector_store.similarity_search(question)
        prompt = f"Answer using only the following:\n{docs}\nQ: {question}\nA:"
        return openai_llm.chat.completions.create(messages=[{"role":"user","content": prompt}])
    result = rag_answer("What is LangSmith?")
    ```

    Here, `@traceable` will ensure the function call is logged as a trace in LangSmith, capturing the model calls inside. The `wrap_openai` makes the OpenAI client log each call. You’d then go to the LangSmith web UI to see the timeline of the retrieval and generation. LangSmith can show token counts and even do evals on outputs if configured.
  * *Features:* It records each **run** (a chain or agent execution) with all inputs/outputs. You can group runs by “project name” for organization. It also supports **evaluation** – e.g., you can label correct vs incorrect answers in the UI or run automated eval functions. Another feature is a **prompt library** to version and test prompts. In short, LangSmith is like the “DevTools” for LangChain apps, helping answer “what did the model see and respond with at each step?”.

* **Helicone:**

  * *Setup:* You integrate Helicone as a proxy in front of your API calls. You can use their hosted cloud (with a free tier) or self-host. Easiest method: set your OpenAI base URL to Helicone’s proxy and include an auth header. For instance, in code:

    ```python
    import openai, os
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = "https://oai.hconeai.com/v1"  # Helicone proxy base
    openai.request_headers = {
        "Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}"
    }
    # Now any openai.ChatCompletion.create calls will go through Helicone
    ```

    The above uses Helicone’s **passthrough proxy**. Helicone will log the request, response, and timing. Alternatively, use Helicone’s SDK or simply route calls via their endpoints as shown. For other providers (Anthropic, etc.), Helicone supports them too with different base URLs or via OpenRouter integration.

  * *Logging & Dashboard:* Once integrated, every call appears in Helicone’s dashboard (requests page). You’ll see for each request: model, prompt (you can choose to log full prompt or just metadata), latency, tokens used, cost, etc. Helicone also aggregates metrics like **requests over time**, **tokens per day**, **errors count**, etc., in a dashboard. The **Top Models** and **Top Users** panels can help identify usage patterns (for instance, which model is costing you most).
    &#x20;*Helicone Dashboard:* Helicone’s dashboard provides real-time monitoring of your LLM usage – requests per period, error breakdown (HTTP 400/500), cost analysis, latency distribution, etc. Developers can quickly spot spikes or issues (e.g., a sudden increase in errors) and drill down into individual logs for details.

  * *Advanced:* Helicone allows tagging requests with custom metadata (like `user_id` or `feature_name`) by adding headers or special fields, so you can filter logs later (e.g., see all requests from a particular user or part of your app). It also supports **caching**: you can enable a cache so identical requests return cached responses, saving tokens (useful during development or if you have repeated prompts). Additionally, Helicone can forward logs to systems like PostHog or your own database, and self-hosters can query the logs SQL database directly.

**How to Choose – LangSmith vs Helicone:**

* **LangSmith** is great when you’re building complex chains/agents (especially with LangChain) and want to trace the *flow* (which tool was called, what intermediate prompt was). It’s focused on **developer debugging** and prompt/version management. It’s less about raw metrics and more about insight into LLM reasoning. It also provides an evaluation framework to systematically test prompt changes.
* **Helicone** is more about **production monitoring**. It’s language/framework agnostic because it just sits as an API proxy. It shines in **aggregated analytics**: if you need to report “We did 10k requests costing \$5 today with median latency 1.2s” or see usage per customer, Helicone does that easily. It’s also simpler to integrate initially (just change API base URL). However, it doesn’t automatically show chain-of-thought traces or tool details – you’d have to log those manually as metadata if you want them.

**Using Both:** It’s possible to use LangSmith for tracing logic and Helicone for overall monitoring simultaneously. For example, you could call OpenAI via Helicone (so Helicone logs it) and also have LangChain log to LangSmith. They complement each other. But if time is short in Buildathon, choose one: use **LangSmith** if you’re iterating on prompt flows and want to quickly debug agent decisions; use **Helicone** if you need to optimize and showcase how your app performs (especially to demonstrate low latency or cost logs).

**Setup Quick Tips:**

* For **LangSmith**, after setting env vars, you can test by a simple trace:

  ```python
  import langsmith
  langsmith.tracing_enabled()  # returns True if LANGSMITH_TRACING true
  from langchain.llms import OpenAI
  llm = OpenAI()  # normal usage in LangChain
  llm.predict("Hello")  # this call will be traced and visible in LangSmith
  ```

  Then visit the LangSmith app, you should see a run with input “Hello” and output. If not, check that `LANGSMITH_API_KEY` is correct and that you called `langsmith.init(...)` if needed (recent versions auto-init from env).
* For **Helicone**, after setting up and making a test call, log into Helicone dashboard (or your self-host) and verify the request shows up. If not, check that you used the correct proxy URL and the `Helicone-Auth` header. Helicone also allows logging without proxy via their SDK (`helicone.log(request, response)` asynchronously) if proxy is not ideal, but proxy is simplest.

**Common Pitfalls & Debugging:**

* *LangSmith not logging:* Ensure `LANGSMITH_TRACING=true` is set **before** running the code. If using Jupyter/Colab, you might need to restart after setting env var. Also ensure your LangChain integration is correct: if you use `langchain.trace` (the older method) or mis-set the project name, runs might not appear where you expect. The LangSmith quick start suggests explicitly wrapping or using context managers if needed. If nothing appears, try forcing a trace: `with langsmith.trace_as("Test run"):` context and run an LLM call inside it. That will ensure a run is recorded even if auto-trace didn’t pick it up due to compatibility issues.
* *Helicone not logging:* The most common cause is forgetting to set `openai.api_base`. If your requests still go to api.openai.com directly, Helicone never sees them. Double-check by intentionally providing a wrong API key – if Helicone is in effect, the error will come from Helicone (and you’d see it in their logs as a 401). If Helicone is working but you don’t see data in dashboard, maybe you used the wrong Helicone API key (it has separate read vs write keys `sk-...` vs `pk-...` for their API, but for proxy either works depending on how they set it up). Use the key from Helicone settings with write access.
* *Data privacy:* By default, Helicone logs full prompts and responses. If your data is sensitive, consider enabling “Omit logs” for content (Helicone has an `Helicone-Cache-Enabled` or other headers to control what to log, or you can choose to only log metadata). LangSmith by default will store prompt data on their servers (if using their cloud) – which may be fine for Buildathon, but keep in mind for real apps. Both tools allow self-hosting for full control (LangSmith has an on-prem or open source variant for enterprise).
* *Overhead:* These tools inevitably add some overhead. LangSmith tracing adds slight latency to log data (usually minimal, and you can sample – by default it might not trace every single call unless enabled). Helicone as a proxy adds a small latency (\~50-100ms typically) which is usually negligible compared to LLM latency, but in tight loops it could add up. If performance is paramount in demo, you might disable them then – but generally the insight gained outweighs a tiny delay. Helicone’s caching can actually improve perceived speed for repeat questions.
* *Interpreting the logs:* In LangSmith, each run may have nested sub-runs (e.g., an Agent run contains Tool runs). Use the UI to expand those. If a chain failed, look for red error markers in the trace. In Helicone, use filters – e.g., filter by `model:gpt-4` to see only those, or sort by latency to find slowest calls. Helicone also can send **alerts** if error rate goes high (some setup needed) – likely overkill for a hackathon, but good to know.

<br>

## 8. Vercel AI SDK + Next.js

The **Vercel AI SDK** is a toolkit for building AI-powered **frontend** apps, especially with Next.js. It provides React hooks and streaming utilities that make it easy to integrate LLM calls into a web UI with support for **Edge Functions** and **Serverless** on Vercel’s platform. Paired with Next.js, you can quickly create chat or completion interfaces that stream tokens to the client.

**Cheat Sheet – Key Components:**

* **Installation:** In a Next.js project, run `npm install ai @vercel/ai` (the package was formerly `@vercel/ai` but might now just be `ai`). Also install any model-specific packages you need, e.g., `@ai-sdk/google` for Google’s PaLM/Gemini, or `openai` if using OpenAI. According to a guide, e.g., `npm install ai @ai-sdk/google @ai-sdk/react zod` (Zod is for schema validation if using structured outputs).
* **Hooks:** The SDK provides **React Hooks** to manage AI interactions:

  * `useChat` – hook to manage a chatbot conversation (messages state, handling user input and streaming assistant responses automatically).
  * `useCompletion` – simpler hook to get a text completion for a given input (e.g., autocomplete or single-turn Q\&A).
  * `useAIStream` (lower-level) – to manually stream responses from an AI API if not using above hooks.
    These hooks abstract away a lot: with `useChat`, for example, you get methods like `append` or `reload` and a `messages` array, and you don’t have to manually call fetch on an API route – the hook does it behind scenes.
* **Providers:** The SDK supports multiple AI providers via a unified interface by using **“Language Model Specification”**. For example, to use OpenAI, you might not need to import anything special – the default fetch could call your internal API route which in turn calls OpenAI. However, the SDK also directly supports calling certain providers from client if needed. E.g., `import { OpenAI } from "ai/providers/openai"` (just hypothetical path) or using config to point at Vercel’s *AI Proxy* (if configured). Vercel also launched an **AI Gateway** which can route requests to various models with one key, but that’s optional.

**Quick Example – Next.js Chat App:**

In a Next.js (App Router) page or component, you can do:

```tsx
'use client'
import { useChat } from 'ai/react'

export default function ChatPage() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    onResponse: (stream) => {
      // optional: handle each streamed token chunk if needed
    },
    api: '/api/chat'  // Next.js API route to handle the request
  })

  return (
    <div>
      <h1>AI Chat</h1>
      <ul>
        {messages.map(m => (
          <li key={m.id} className={m.role}>{m.content}</li>
        ))}
      </ul>
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} placeholder="Say something..." />
        <button type="submit">Send</button>
      </form>
    </div>
  )
}
```

And in `/api/chat` (Next.js API route):

```ts
import { OpenAIStream, StreamingTextResponse } from 'ai'
import { Configuration, OpenAIApi } from 'openai-edge'  // using edge runtime
export const runtime = 'edge'  // run this on Vercel Edge

const config = new Configuration({ apiKey: process.env.OPENAI_API_KEY })
const openai = new OpenAIApi(config)

export async function POST(req: Request) {
  const { messages } = await req.json()
  const response = await openai.createChatCompletion({
    model: 'gpt-3.5-turbo',
    stream: true,
    messages
  })
  const stream = OpenAIStream(response)  
  return new StreamingTextResponse(stream)
}
```

A few things to note:

* `useChat` on the client automatically handles form input and sending a POST to `/api/chat` with the conversation messages (it uses the `api` field we provided) so we didn’t have to manually wire up fetch – `handleSubmit` does it. The `messages` it maintains include both user and assistant messages, each with an `id`, `role`, and `content` property.
* On the server API route, we use **openai-edge** SDK which is compatible with Edge runtime. We set `runtime='edge'` so this runs on Vercel’s edge (low latency global). This code streams the completion by calling OpenAI with `stream: true`, then wrapping the response in an `OpenAIStream` (from the Vercel AI SDK) and returning a `StreamingTextResponse` – which the Vercel AI client knows how to consume gradually.
* The net effect: as the OpenAI API streams tokens, the Next.js API route yields them chunk by chunk to the client. The `useChat` hook receives those via server-sent events and updates `messages` state in real-time, giving a smooth streaming chat UX without you writing extra code.

**Key benefits & concepts:**

* **Edge Functions:** By running on `runtime: 'edge'`, you get extremely fast response start times (\~10ms) because the function is deployed globally on Vercel’s Edge network. And streaming ensures the user sees output token-by-token with low latency.
* **Generative UI Components:** The Vercel AI SDK includes some React components (like a `<ChatBot>` component or utility components to format messages) but the primary is hooks. It also has support for **AI-generated UI** with React Server Components (RSC): e.g., `useAssistant` can render React components from LLM output (experimental). This is advanced – basically the model can output a JSON describing a UI and the SDK can render components. It’s a novel idea (e.g., model says to show a graph, and a `<Chart>` component is rendered with data). But likely not needed in Buildathon unless you do something fancy.

**Setup Instructions:**

* **Creating Next.js App:** If starting fresh, run `npx create-next-app@latest --typescript`. Then add the dependencies. Possibly start from a template: Vercel provides open-source AI templates (like a chat template using this SDK) – you can literally clone those to get a working example quickly.
* **Environment Variables:** Put your API keys in Next.js env (`.env.local`) and ensure they’re accessible (for edge functions, you use process.env normally, just mark them in next.config if needed). The OpenAI Edge SDK doesn’t read from env automatically, we pass it in `Configuration`.
* **Vercel Deployment:** If you deploy to Vercel, add the env vars in project settings. The edge function should work out-of-the-box. The streaming uses web standards, so no special setup. Test locally (`npm run dev`) and you should see streaming in your terminal and on page.

**Common Pitfalls & Debugging:**

* *CORS/URL issues:* If you use the hooks and API route approach, keep them on the same domain to avoid CORS issues. The `api` option in hook can be just `'/api/chat'` (relative path) which is simplest. Ensure correct `export function POST` signature for App Router (or use pages/api for older approach). A 404 or CORS error likely means the endpoint path is wrong or not caught by Next.
* *Not streaming:* If you see the response only after completion, likely the streaming isn’t hooked up right. Make sure you return `StreamingTextResponse` (or in older usage, use Next.js `new Response(stream)` with proper headers). Also ensure `model: 'gpt-3.5-turbo'` is a streamable version; if not using the June 2023 API with function-calling, prefer the `-0613` suffix models. The OpenAIEdge’s `createChatCompletion` returns a `FetchResponse` that you must convert with `OpenAIStream`. Not doing so means you just wait for it to end.
* *Edge function limitations:* Edge environment doesn’t support all Node modules (no filesystem, etc.). But since we just call external APIs, that’s fine. If you need to do vector DB retrieval in the same request, you might not be able to unless that DB has an HTTP endpoint (Edge can fetch external URLs). Or you run as Node runtime instead of edge (then it might be slower). For Buildathon, maybe avoid heavy server-side stuff in the streaming route; keep it mostly calling the LLM.
* *Token flicker/double printing:* The `useChat` hook handles stream nicely, but if you manually append messages in onResponse, you might get doubling. Usually, you don’t need to manually update state – the hook does it. Just ensure you feed the `messages` it provides back into the AI call (which it does by default). If customizing, maintain message history.
* *Compatibility:* The Vercel AI SDK is evolving – function names might change. Always check latest docs for correct import paths (`ai/react`, `ai` for OpenAIStream, etc.). If you get build errors, ensure versions match.
* *Next.js specifics:* If using App Router, remember to export components as default if using in routing. Also, if you want to disable SSR for the chat component (maybe you do, maybe not), you can keep `'use client'` as in example to make it purely client-side interactive. SSR isn’t very relevant for chat (you usually don’t pre-render chats on server).

Overall, **Vercel AI SDK + Next.js** lets you deliver a polished web demo quickly: you get nice UI and fast performance with minimal effort on streaming. Combined with Vercel’s hosting, you can show live logs (Vercel’s edge functions logging or integrate Helicone in the API route to log usage). Also, consider using **Vercel’s `ai` package functions for built-in rate limiting** (it has features like built-in caching and rate limiting for OpenAI API). But for a hack demo, likely not needed unless you anticipate heavy load.

<br>

## 9. Replit Deployments

**Replit** is an online IDE that can run and host apps directly from your browser. Replit Deployments offer a quick way to turn your project into a **public web app or service** with minimal DevOps. This is useful in a Buildathon to share your demo or host an API for your project.

**Cheat Sheet – Deployment Types:**
Replit has multiple deployment options to cater to different needs:

* **Autoscale Deployment:** Runs your app with autoscaling on demand – resources ramp up with usage and down when idle. Good for unpredictable traffic, as it can handle spikes (within limits). Pricing may be usage-based.
* **Static Deployment:** Serves static frontends (HTML/CSS/JS with no backend) cheaply. Great if your demo is purely client-side or uses only external APIs.
* **Reserved VM (Always On):** Gives you a fixed VM with guaranteed RAM/CPU so your app is always running and responsive. Useful for long-running bots, Discord bots, or if you need websockets or background tasks. This is like a persistent server (but costs more).
* **Scheduled Deployment:** Runs your code on a schedule (cron jobs). Probably not needed for demo (it’s for e.g. daily tasks).

For a typical web demo, **Autoscale** is often recommended: it’s easy and cost-effective (it might spin down when not in use). If using Replit’s free tier, note that always-on might not be available, but the deployment still can be accessed via a generated URL when awake.

**How to Deploy:**

1. **Build your app on Replit:** Either create a new Repl and code in their IDE, or import a GitHub repo. Ensure your app listens on the correct port (Replit usually expects you to use port 8080 or the port in `process.env.PORT`) if it’s a web server. For example, a simple Flask app or Node Express app.
2. **Hit the Deploy button:** In Replit, there is a “Deploy” tab. Choose **Deployments** and then “New Deployment”. It might ask you to choose type – the system will suggest one. If it’s a web app, it often picks Autoscale by default.
3. **Configure and Launch:** Give it a name, pick the branch (main) to deploy. Add any required **Secrets** (env vars) in the Secrets section (these will be available during deployment). Then deploy. Replit will build a snapshot and host it.

Once deployed, you get a **deployment URL** (like `https://<appname>.<username>.repl.co` or a similar domain). Share this with judges or teammates. Replit provides a web UI to monitor logs and status of your deployment. You can also set up a **custom domain** if you want (Replit allows linking a domain in settings).

**Example:** If you have a Streamlit app from section 13, you can deploy it on Replit. If Streamlit runs on port 8501 by default, Replit’s environment sets `PORT`, so you might run `streamlit run app.py --server.port $PORT` in the Replit deployment script. Similarly for Flask:

```python
import os
from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello(): return "Hello from Flask!"
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv('PORT', 8080))
```

When you deploy this, Replit will ensure it’s accessible at the deployment URL.

**Live Logs & Debugging:** In Replit’s Deployments interface, you can see **live logs** of stdout/stderr of your app (it’s similar to Heroku logs). Use this to debug runtime errors or see print outputs. You can also enable **Web embedding** to embed the app in an iframe in e.g. Devpost if needed (Replit allows you to embed the output).

**Collaboration:** One neat thing – if your team members are coding on Replit, you all can live-edit the code (multiplayer editing) and see changes. But note: the deployed snapshot won’t update until you redeploy. There is a “auto-deploy on push” setting if connecting to Git.

**Common Pitfalls:**

* *Deployment vs Repl run:* Running in the Replit IDE vs deploying are separate. The IDE “Run” runs it in development (and gives a `.repl.co` URL as well, but ephemeral). Deployment creates an immutable snapshot. If your code works in the IDE run but fails in deployment, check if environment variables in deployment are set (the dev run might allow certain things that the deployment environment doesn’t, like filesystem persistence differences).
* *Persistence:* In most deployments, the file system is read-only or ephemeral. For example, if your app writes to disk, those changes might not persist across restarts (especially in autoscale, each instance is fresh from snapshot). Use a database or Replit’s built-in DB if you need to store state. Or for hack demos, simply keep things in memory or accept non-persistence.
* *Sleep and cold starts:* If using Autoscale deployment, note that if no one accesses the app for a while, it might spin down. The first request then experiences a cold start (a few seconds of delay while container starts). This is usually fine; just be aware when demonstrating – if it’s been idle, give it a moment to wake. You can ping it just before judging to keep it warm. Reserved VM deployments don’t sleep but require you to have an always-on Repl (which typically requires a paid plan).
* *Resource limits:* Free Replit deployments have limits on RAM/CPU. If your LLM app is heavy (say running local model), it might not fit. Replit’s paid plans allow more. For calls to cloud APIs, memory isn’t big issue, but if you accumulate too much in memory (like storing huge embeddings in memory), you could hit memory cap and crash. Monitor the memory usage shown in Replit. Also, infinite loops or high CPU can trigger the Repl to be killed. Use efficient logic, and break or background heavy tasks if possible.
* *WebSocket or interactive issues:* Some frameworks (like very interactive ones or those requiring sticky sessions) might not play well with autoscale (since each request could hit a different instance). But for hack demos, likely you don’t have multi-instance scaling with the small traffic. If you do use WebSockets (for streaming output to client), ensure using a supported method – Replit deployments support WebSockets on Reserved or Autoscale (I believe they do, as underlying is likely a container on NixOS). If any issue, consider falling back to SSE (server-sent events) or just periodic polling.
* *Deployment config:* Check the **Replit config (replit.nix or pyproject.toml)** to ensure all dependencies are listed so that the deployment environment has them. Sometimes a package works in IDE due to caching but the deployment build might miss it. Open the “Shell” in Replit and try a fresh install to see if anything is missing.

**Benefits for Buildathon:** Using Replit means you avoid dealing with AWS/GCP for hosting – it’s all in-browser. It’s especially great for showing code and running it in one place. Many hackathons actually encourage Replit for quick demos. Additionally, Replit has an AI assistant (“Ghostwriter”) which could help code, but main focus here is deployment.

Remember, for a public demo, you can just share the deployment URL. If your app has an API (like a backend endpoint), you could also call it from elsewhere; Replit’s deployments essentially give you a web server accessible to the internet.

<br>

## 10. Supabase Vector (pgvector in Postgres)

**Supabase** is an open-source Firebase alternative built on PostgreSQL. It supports the **pgvector** extension for storing and querying vectors (embeddings) directly in Postgres. This turns your Supabase database into a vector search engine. Using Supabase Vector, you can implement RAG (Retrieval-Augmented Generation) by storing document embeddings and performing similarity search via SQL.

**Cheat Sheet – Using pgvector in Supabase:**

* **Enabling pgvector:** Supabase now has pgvector enabled by default on new projects (or at least available). If not, go to your Supabase project dashboard -> Database -> Extensions, find "vector" and enable it. You can also do `CREATE EXTENSION vector;` in an SQL query. This loads the pgvector extension which defines a new column type `VECTOR`.
* **Creating a Table with Vector Column:** Decide your embedding dimensionality (e.g., 1536 for OpenAI text-embedding-ada-002). Create a table:

  ```sql
  CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    content text,
    embedding vector(1536)
  );
  ```

  This will store content and its embedding. You might also store metadata or chunk info. (In Supabase, you can use their SQL editor or the JS/py SDK to execute this).
* **Inserting Embeddings:** Use your LLM or embedding API to get embedding vectors (array of floats). Then insert via parameterized query or the Supabase client. Example in JavaScript:

  ```js
  // using Supabase JS SDK
  const { data, error } = await supabase.from('documents')
    .insert({ content: 'Hello world', embedding: embeddingArray });
  ```

  The array should be a float32 array or list of numbers. The Supabase JS client will automatically convert it to the Postgres vector type literal (e.g., `'[0.12, 0.53, ...]'`). In SQL, you could also explicitly write `INSERT ... embedding = '[0.12, 0.53, ...]';`. As shown in docs, you can use the Supabase Transformers library in JS to generate embeddings then directly store (they gave an example with Transformers.js generating and storing in one go).
* **Creating an Index:** For faster similarity search at scale, create an **index** on the vector column. Postgres pgvector supports approximate indexes like IVF (Ivfflat) or HNSW. Example:

  ```sql
  CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
  ```

  This creates an IVF index using cosine distance. `lists=100` is a tuning parameter (more lists = faster search but slower insert). Alternatively, `vector_l2_ops` for Euclidean or `vector_inner_product_ops` for dot-product. If your embeddings are normalized, cosine is good. Do this after you have some data or up front. Without an index, similarity search will be brute force (okay for small scale).
* **Querying (Vector Similarity Search):** Use the `<->` operator to compute distance between vectors. For example, to get 5 most similar:

  ```sql
  SELECT id, content, embedding
  FROM documents
  ORDER BY embedding <-> '[0.1, 0.2, ... , 0.05]'  -- your query vector
  LIMIT 5;
  ```

  The `<->` operator by default does Euclidean distance, unless you created index with cosine\_ops (then `<->` interprets as cosine distance). The result will be the nearest neighbors. If you want the distance values, you can also select `embedding <-> '[..]' AS distance`.
  In Supabase JS client, you can perform this via `.rpc()` calling a Postgres function or using `.select().order('embedding', { foreignTable: '...', ascending: true, ...})` – but Supabase hasn't yet provided a high-level method for vector search as of now, so the common approach is to create an RPC (remote procedure). For example, define a SQL function:

  ```sql
  CREATE FUNCTION match_documents(query_vec vector(1536), match_count int)
  RETURNS TABLE(id bigint, content text) AS $$
  SELECT id, content
  FROM documents
  ORDER BY documents.embedding <-> query_vec
  LIMIT match_count;
  $$ LANGUAGE SQL IMMUTABLE;
  ```

  Then in your code: `supabase.rpc('match_documents', { query_vec: embeddingArray, match_count: 5 })`. This will return the rows. (Supabase is considering adding a direct `.vectorSearch()` API in future).

**Example – Putting It Together (Pseudo-code):**

Let’s say you have a bunch of FAQ texts and want to answer user questions by retrieving relevant answers:

* **Setup:** All FAQ entries are embedded (using OpenAI embeddings) and stored with pgvector.
* **Query flow:** When user asks a question, you embed the question with the same embedding model. Then do the SQL query to get top 3 similar FAQ entries. Then feed those to the LLM to generate a final answer (that’s the RAG pattern).

Pseudo-code (Node):

```js
// Given a user question:
const question = "How do I reset my password?";
// 1. Get embedding for the question (using OpenAI embedding endpoint via Supabase or OpenAI SDK)
const embedding = await openai.createEmbedding({input: question, model: 'text-embedding-ada-002'});
// 2. Vector search in Supabase for relevant docs
let { data: matches, error } = await supabase.rpc('match_documents', {
  query_vec: embedding.data[0].embedding, match_count: 3
});
if (error) console.error(error);
const context = matches.map(m => m.content).join("\n---\n");
// 3. Ask LLM with the context
const prompt = `Answer the question using the following FAQs:\n${context}\nQ: ${question}\nA:`;
const completion = await openai.createChatCompletion({
  model: 'gpt-3.5-turbo', messages: [{ role: 'user', content: prompt }]
});
console.log(completion.data.choices[0].message.content);
```

That shows Supabase vector search in action. The heavy lift was done by Postgres (finding similar FAQ content). The LLM just uses it to answer, reducing hallucinations.

**Setup Instructions:**

* **Supabase Project:** If you haven’t, create a free Supabase project. Use SQL editor to create tables or use their Table editor UI. Enable vector extension if needed as described.
* **Supabase Client:** Use the official client libraries (available for JS/TS, Python, etc.). Provide the Project URL and anon API key. Note: for server-side usage, you might want to use the service\_role key for inserts if you bypass Row Level Security (RLS). Supabase by default has RLS on with no open access. For prototyping, you can disable RLS on the table or write policies to allow the operations you need. Simplest: disable RLS on the documents table so you can insert/select freely (but careful with exposing anon key if so).
* **Dimension mismatch:** Make sure you use the same embedding dimension consistently. If you try to insert vector of wrong length, Postgres will error. The `vector(1536)` type will enforce length. Also ensure the query vector you pass in has exactly that length.
* **Data size considerations:** If you have thousands of embeddings, create the IVF index early to speed up. Without it, the `<->` search will linearly scan – that might be slow beyond a few K rows. With IVF, query is fast (ms range) but note: PG’s IVF is approximate by default. You can tune recall via `probes` when querying (e.g., `SET ivfflat.probes = 10;` for higher accuracy at cost of speed). For hackathon scale (hundreds to low thousands of docs), brute force might be fine and simpler (just ensure performance is okay in tests).

**Common Pitfalls & Debugging:**

* *pgvector not installed:* If queries error like “operator does not exist: vector <-> vector”, it likely means the extension isn’t enabled. Ensure `vector` extension is created on your database.
* *Inserting vectors format:* If you try to insert as a string incorrectly, or as JSON, it may fail. The expected literal format in SQL is `'[0.1, 0.2, ...]'` (square brackets). Supabase client should handle JS array -> Postgres vector automatically. In Python, the Supabase py client might need a list of floats (which it serializes to that format). If issues, explicitly convert to str(`[${floats.join(",")}]`).
* *Distance metric confusion:* `<->` by default does Euclidean. If you want cosine similarity but forgot to use `vector_cosine_ops` in index, your results might be off. Workaround: if all embeddings are normalized, Euclidean ordering is equivalent to cosine ordering (monotonic relationship). But if not, consider either normalizing them or using inner product and treat max dot product as similarity. Alternatively, compute cosine distances on the client after retrieving a few candidates. The simplest is: use OpenAI embeddings (they’re already normalized to length 1 roughly), and use Euclidean – should work fine as is. Or explicitly do `ORDER BY embedding <-> '[..]'` after having set index to cosine, that will treat `<->` as cosine distance due to ops context.
* *Supabase Rate limits:* Supabase free tier allows a certain amount of requests per second and a monthly quota. Vector search is just a SQL query, it counts towards that. For hackathon usage, likely fine. If you suddenly spam many calls, you could hit rate limits (you’d see 429 errors). In that case, backoff or upgrade plan. Also, each embedding vector is \~6KB if dimension 1536 of float32 (4 bytes each) – a thousand vectors is \~6MB, which is okay, but tens of thousands might bloat DB quickly. Manage your data (maybe store fewer or use smaller dimension model if possible, e.g., 384-dim MiniLM embeddings via HuggingFace).
* *Integration with other tools:* If you use LangChain, note that it has a SupabaseVectorStore integration which basically does what we described under the hood (embedding and storing in table) but for simplicity, our direct approach is fine. For debugging queries, use Supabase SQL editor to run a sample query and see if you get results. If nothing returns, either the vector didn’t insert correctly or the query vector isn’t similar enough (maybe try without limit to see if anything comes out with a distance, or order ascending to see smallest distance = most similar).
* *Security:* If building an API that exposes vector search directly, be careful with SQL injection (Supabase RPC avoids direct injection by treating inputs safely, but if constructing SQL manually, always parameterize). If you enable RLS, consider adding policies like “user can select from documents if user\_id matches” (in case multi-user scenario). For hackathon demo, you might not need RLS complexity – just keep it open or internal.

<br>

## 11. Cloudflare Workers AI + Vectorize

**Cloudflare Workers AI** is a new offering allowing you to run AI models (or call AI APIs) directly within Cloudflare’s serverless edge environment. Paired with **Vectorize**, Cloudflare’s globally distributed vector database, you can build AI apps that run at the edge, with data stored at the edge for fast vector searches. Essentially, Cloudflare is bringing compute and vector storage closer to users, potentially reducing latency significantly for AI tasks.

**Cheat Sheet – Getting Started:**

* **Workers AI Basics:** Cloudflare Workers are JavaScript/TypeScript functions that run on Cloudflare’s edge network (200+ locations). Workers AI introduces an `env.AI` binding that lets you run model inference (they have some built-in models like a smaller text model, and allow calling external ones, possibly via their AI Gateway). It supports running open-source models on Cloudflare’s infrastructure (like certain medium-sized models) without contacting external APIs. For example, they mention a built-in embedding model `@cf/baai/bge-base-en-v1.5`.
* **Vectorize Basics:** Cloudflare Vectorize is a managed vector DB where indexes are replicated globally. You interact with it through the Worker via an `env.VECTORIZE` binding in the worker that exposes a client with methods like `.upsert()` and `.query()`. You create an index (with a given dimensionality and metric) using the wrangler CLI, bind it to your Worker, then can put vectors and do similarity queries all from within the Worker code.

**Setting up:**

1. **Wrangler CLI:** Install Cloudflare’s wrangler (`npm install -g wrangler`). Log in to Cloudflare account (`wrangler login`).
2. **Create a Worker project:** e.g., `wrangler init my-ai-worker`. Choose “Hello World” template as guide suggests. In `wrangler.toml`, you’ll add bindings for AI and Vectorize. It might look like:

   ```toml
   name = "my-ai-app"
   main = "src/index.ts"
   compatibility_date = "2023-10-01"
   [[vectors]]
   binding = "VECTORIZE"
   index_name = "embeddings-index"
   [[ai]]
   binding = "AI"
   # no model specified here means we'll specify at runtime
   ```

   This tells Cloudflare to bind your Worker to the Vectorize index named "embeddings-index", and to enable Workers AI for model running with the binding name "AI".
3. **Create a Vector Index:** Use wrangler or Cloudflare dashboard:
   `wrangler vectorize create embeddings-index --dimensions=768 --metric=cosine`. This sets up an index with given dimensions. If success, wrangler will output the binding config to add (we did that above).

**In your Worker code (`src/index.ts`):**
Use `env.AI` to run embeddings and `env.VECTORIZE` to store/query:

```ts
import { EmbeddingModel } from "@cloudflare/ai"; // hypothetically, to get type for model id

export default {
  async fetch(request: Request, env: Env) {
    // For simplicity, suppose request contains a JSON with { text: "..." } to upsert.
    if (request.method === "POST") {
      const { text, id } = await request.json();
      // 1. Generate embedding using Cloudflare's built-in embedding model
      const embedding = await env.AI.run("@cf/baai/bge-base-en-v1.5", text);
      // 2. Upsert vector into index
      await env.VECTORIZE.upsert([
        { id: id, values: embedding.data[0], metadata: { text } }
      ]);
      return new Response("Inserted", { status: 200 });
    }
    // GET for query
    if (request.method === "GET") {
      const url = new URL(request.url);
      const query = url.searchParams.get("q");
      const queryEmbedding = await env.AI.run("@cf/baai/bge-base-en-v1.5", query);
      const matches = await env.VECTORIZE.query(queryEmbedding.data[0], { topK: 3 });
      return Response.json(matches);
    }
    return new Response("Use GET or POST", { status: 400 });
  }
}
```

This worker does both insertion and querying. On a POST, it expects JSON with text and an id, it uses `env.AI.run` with an embedding model to vectorize the text (the result is likely an object with `.data` containing the vector), then calls `env.VECTORIZE.upsert()` to store it. On a GET with `?q=...`, it similarly embeds the query and calls `query()` to get nearest neighbors. `env.VECTORIZE.query` will return an array of matches with their ids, vectors, and metadata.

**Deployment:**
Use `wrangler deploy` to publish this worker. It will output a URL like `https://my-ai-app.<your-subdomain>.workers.dev`. You can then test it: POST some texts to store, then GET query. The beauty is – this runs on Cloudflare edge, so both the vector DB and model inference are at the edge, making it very fast globally.

**Common Pitfalls & Notes:**

* *Model availability:* The `@cf/baai/bge-base-en-v1.5` is an open model for embeddings. Cloudflare likely has a set of such open models (maybe also some text generation ones). Check their docs for model identifiers (like stable diffusion or others might be available). If you try a model name that’s not available or you’re not authorized, `env.AI.run` might throw error. Use models listed in Cloudflare’s llm resources (they provided `llms.txt` listing models).
* *Data locality:* Vectorize is global – but note, it might not be *strongly consistent global*. They likely replicate data to multiple regions. For small hack usage, not a big issue, but be aware of eventual consistency if documented. The advantage is queries from anywhere are fast, as the data is near.
* *Limits:* Cloudflare Workers free tier has limits (100k requests/day, memory 128MB etc.). Workers AI and Vectorize in beta might have their own quotas (like limited vector count, certain model usage limit). Check if any env variable needed for Workers AI usage (maybe Cloudflare gives some token for beta). If you hit an error like “AI binding is not allowed”, ensure your account has access (the product might still be in limited beta; assume by 2025 it’s open).
* *Testing locally:* `wrangler dev` can simulate the worker, but it might not run the AI bindings locally. Likely you have to test by deploying to see actual model inferences (since those run on cloud). Use lots of logging (console.log in worker, wrangler dev will show logs, or `wrangler tail` to stream logs from deployed worker).
* *Performance:* The appeal of this stack is speed. In practice, a vector query on Vectorize is said to be low-ms and model inference at edge might be a bit slower than calling OpenAI’s API but still decent (embedding model is smaller than OpenAI’s Ada). If needed, you can also call OpenAI from worker using fetch (but that loses the “no external calls” advantage). The above example used Cloudflare’s on-platform embedding to avoid external API. Also, Workers can run concurrently lots of requests, but each is short-lived (50ms CPU time free, more requires paid). Model inference might take >50ms, but Cloudflare likely charges for that usage. For hack, fine.
* *Vector size mismatches:* If you declare index dimension 768 and use OpenAI’s 1536-dim embedding, that’s wrong. Ensure to align dimension. The example used 768, which corresponds to BGE base. If you needed 1536 (OpenAI Ada), you’d create index with 1536 dims and then either call OpenAI or find a similar model on CF (I think BGE-large might be 1024? Not sure, but let's say use what’s available).

In summary, Cloudflare Workers AI + Vectorize is cutting-edge and perhaps more experimental, but demonstrates how you can deploy a full-stack RAG app fully on the edge cloud. If it’s stable enough during the Buildathon, it could impress with its low latency and the fact that no separate DB or GPU server is needed – Cloudflare does it all.

<br>

## 12. MongoDB Atlas Vector Search

MongoDB Atlas (the cloud service for MongoDB) has introduced **Vector Search** capabilities in its full-text search engine. This means you can store embeddings in MongoDB and use the Atlas Search `$vectorSearch` operator (or earlier, `$knnBeta`) to find similar vectors within a collection. This leverages Lucene under the hood (since Atlas Search is built on Lucene) to do approximate nearest neighbor search.

**Cheat Sheet – How to Use Atlas Vector Search:**

* **Data Model:** Typically, you store your documents with an embedding field. For example:

  ```js
  {
    _id: ...,
    content: "Some text...",
    embedding: [0.123, -0.045, ... , 0.879]  // array of floats
    // plus any metadata fields you want
  }
  ```

  The field can be a **dense vector** of a fixed size. As of MongoDB 7.0+, they support float32 arrays up to certain length (I believe up to 1024 in current GA, and more in preview). If your dims > 1024, you may need to enable a preview feature or reduce dim (or use two vectors). Check Atlas docs – by 2025, likely 1536 dims (OpenAI) is supported.
* **Creating a Search Index:** You must create an **Atlas Search index** on the collection to use vector search. In the Atlas UI, go to your cluster, select Search Indexes, “Create Index”. JSON config example:

  ```json
  {
    "mappings": {
      "dynamic": false,
      "fields": {
        "embedding": { "type": "knnVector", "dimensions": 1536, "similarity": "cosine" }
      }
    }
  }
  ```

  This defines that `embedding` field is a knnVector with given dimensions and using cosine similarity. Set dynamic to false to only index specified fields (often safer). You can still combine with other fields for filtering, etc., in queries.
* **Querying with \$vectorSearch:** In Mongo shell or via driver, you use an aggregation pipeline with `$search` stage (Atlas Search) to perform vector similarity:

  ```js
  db.collection.aggregate([
    {
      $search: {
        index: 'myVectorIndex',
        knnBeta: {
          vector: [0.12, -0.34, ...],   // query embedding array
          path: "embedding",
          k: 5
        }
      }
    },
    { $project: { content: 1, score: { $meta: "searchScore" } } }
  ])
  ```

  This uses the older `knnBeta` operator (the newer syntax might be `$vectorSearch` with similar parameters). As of 2024+, `$vectorSearch` became the official operator:

  ```js
  $search: {
    index: 'myVectorIndex',
    vector: {
      query: [0.12, -0.34, ...],
      path: "embedding",
      k: 5
    }
  }
  ```

  (Double-check the exact JSON, it might be `vector: { path, query, k }` as in docs). This will return top 5 documents, and you can use the `$meta: "searchScore"` to get similarity scores. Filtering: you can combine with compound queries or add a `filter` inside vector search (Atlas Search allows a `filter` field alongside knnBeta to e.g. only search within certain category). That is powerful: you can do **hybrid search** (text + vector) easily, by having a compound query with both vector and text conditions.
* **Using via Drivers:** The `$search` stage is only available through the Atlas cluster (not in the local Mongo unless running Enterprise with search enabled). In code, you must run an aggregation. In Python, e.g.:

  ```python
  pipeline = [
    {"$search": {
        "index": "myVectorIndex",
        "vector": {
          "path": "embedding",
          "query": query_embedding,
          "k": 5
        }
      }},
    {"$project": {"content": 1, "_id": 0, "score": {"$meta": "searchScore"}}}
  ]
  results = collection.aggregate(pipeline)
  for doc in results:
      print(doc["content"], " score:", doc["score"])
  ```

  Ensure you use the correct index name (if you didn't name it, default might be "default"). If you get an error that \$search is not allowed, ensure you're connected to Atlas (not a local instance) and have search index created.

**Example Use Case:** Storing product descriptions embeddings to recommend similar products. Steps: embed each product description with a model and store vector. Then for a given product, use its vector as query to find similar. Or for question answering, store knowledge snippets, then get query embedding and find relevant pieces in Mongo, then feed to LLM. Very similar to the Supabase example but using Mongo.

**Setup & Observability:**

* Use MongoDB Atlas UI to monitor query performance. The search stage has profiling – you can see how long the vector search took. For large data, they likely use HNSW under hood via Lucene, which is very fast. The `k` parameter is the number of neighbors to retrieve – set as needed.
* Note that the `searchScore` is a similarity measure. If using cosine, higher = more similar (score is often cosine similarity in range 0-1). If using Euclidean (not typical for normalized embeddings), lower distance might be transformed to a higher score. But likely you'll use cosine.
* If you need to filter by metadata (like only show results from a specific user or date range), you can include a `filter` in the \$search stage. For example:

  ```json
  knnBeta: { path: "embedding", query: [...], k:10, filter: { category: "tech" } }
  ```

  or using compound:

  ```json
  $search: {
    "compound": {
      "must": [{ "text": {... some text query ...} }],
      "should": [{ "vector": { path: "embedding", query: [...], k: 10 } }]
    }
  }
  ```

  which gives a hybrid scoring.

**Common Pitfalls:**

* *Precision vs Performance:* There's an internal parameter called `k` and `knnBeta` by default uses HNSW which may not return exact nearest neighbors if you have a very large dataset. You can configure **recall vs performance** via the index or query options (in older Atlas, they had a `filter` trick to do re-scoring or allowed setting `alpha`). By 2025, this might be abstracted. If you notice results not very accurate, perhaps increase `k` and then post-filter top n by actual distance if needed. For moderate data sizes, it should be fine.
* *Data Ingestion:* There’s currently no direct vector ingestion tool in Atlas; you have to handle computing embeddings. But perhaps they introduced integration with their function or a Python library (some blog posts show using `pymongo` with OpenAI API to batch insert). Watch out for document size – MongoDB docs have 16MB limit. 1536 floats = \~6KB, so fine. Just don’t store huge text and huge vectors in one doc beyond that.
* *Driver support:* Ensure you use an updated driver that supports Atlas Search queries. Most drivers treat `$search` as just another stage, so it's fine. But some older ORMs might block it because it's not a recognized stage. If using Mongoose in Node, you may need to use `Model.aggregate()` not the Mongoose query helpers (since Mongoose might not have direct support for `$search`).
* *Cost:* Atlas Search is an add-on to clusters; M0 (free tier) does support search on limited data, but check quotas (maybe 500MB index on free?). If your data is small, free tier might suffice. If not, you might need at least M10 cluster (which costs). For hackathon, you could possibly use the free tier with a small dataset. Monitor the search usage; heavy search queries could count towards Search Ops (which are billed after a certain free allowance). But likely okay for demo scale.
* *Security:* If exposing this via an API, note that by default you should not let random user provide raw vector that goes directly into \$search without validation (someone could craft a vector that breaks something, though not likely as it's numbers). Also, \$search stage cannot be run with just any privileges - it needs search role. If using an Atlas App Service or something, ensure proper roles. If building in backend, you have the credentials anyway.

**When to pick Atlas vs others:** If your stack already uses MongoDB (for your app’s non-LLM data), adding vector search can simplify architecture (no extra vector DB needed). It’s also convenient if you want to combine text search and vector search easily. It might not outperform specialized vector DBs for huge scale, but for moderate and for easily filtering, it’s very capable. And it’s fully managed in Atlas, so you don’t run separate infrastructure.

<br>

## 13. Streamlit

**Streamlit** is a Python framework for rapidly creating data web apps with minimal code. It’s excellent for demoing machine learning or data science projects by making an interactive UI in a few lines. For LLM demos, you can use Streamlit to build a simple chatbot interface, or a form where user inputs some text and results are displayed. The focus is on speed of development rather than fine-grained HTML control.

**Cheat Sheet – Building a Streamlit App:**

* **Installation:** `pip install streamlit`. Then you create a script, e.g., `app.py`. Running `streamlit run app.py` starts a local server (default [http://localhost:8501](http://localhost:8501)).

* **Basic usage:** Use Streamlit functions to layout elements:

  * `st.title("My LLM Demo")` – big heading.
  * `st.text_input("Enter your question:", key="question")` – text box.
  * `st.button("Submit")` – a button.
  * `st.write(data)` – display text, dataframe, or any object nicely (markdown interpretation, etc.).
    Streamlit apps run top-to-bottom on each interaction (re-run concept). Use **state** via `st.session_state` to store variables across runs (like conversation history).

* **Example – Q\&A App:**

  ```python
  import streamlit as st
  import openai

  st.title("🔎 Document Q&A Bot")
  query = st.text_input("Ask something about the documents:")
  if st.button("Submit") and query:
      # call LLM (OpenAI example)
      response = openai.Completion.create(prompt=f"Q: {query}\nA:", engine="text-davinci-003", max_tokens=100)
      answer = response.choices[0].text.strip()
      st.write(f"**Answer:** {answer}")
  ```

  This will show a text box and a submit button. When clicked, it calls OpenAI API and shows the answer. Notably, this code doesn’t handle reading documents – that would require either embedding search as in previous sections or fine-tuning. But as a minimal example, it works (though answers might hallucinate). You can improve by retrieving relevant text and including it in prompt. The key point is how simple the UI part is.

* **Displaying Outputs:** Streamlit has many built-ins: `st.markdown("**bold** text")` for Markdown, `st.code(code_string, language='python')` to show code nicely, `st.image(image_array)` to show images (like DALL-E results). For LLM demos, you might show a table of retrieved docs: e.g.,

  ```python
  for doc in top_docs:
      st.markdown(f"* {doc[:100]}...")
  ```

  to list snippet of each. You can also use `st.expander` to hide long text behind a clickable expander. For example:

  ```python
  with st.expander("See full document context"):
      st.text(long_text)
  ```

* **Live updates:** If you want to stream tokens as they come (for example, from OpenAI’s streaming completion), Streamlit is a bit tricky because it doesn’t natively provide streaming in the same way. But you can simulate by incrementally adding text in a loop and calling `st.empty()` or placeholders. For instance:

  ```python
  placeholder = st.empty()
  partial_answer = ""
  for chunk in stream_from_openai(prompt):
      partial_answer += chunk
      placeholder.markdown(partial_answer + "▍")  # ▍ as cursor
  placeholder.markdown(partial_answer)
  ```

  This approach updates the placeholder text on each chunk. It works decently to show streaming output. Ensure to flush Python output if needed (but Streamlit typically deals with its own rendering). Keep in mind each loop iteration triggers a re-run – but because we’re in same button click, maybe not. Streamlit might not be perfectly built for token-level streaming, but this hack is known to work.

* **State and Chat History:** Use `st.session_state` to remember past messages in a chat app:

  ```python
  if "history" not in st.session_state:
      st.session_state.history = []
  user_msg = st.text_input("You:", key="user_input")
  if st.button("Send"):
      st.session_state.history.append({"role":"user", "content": user_msg})
      # get response
      assistant_msg = get_chat_response(st.session_state.history)
      st.session_state.history.append({"role":"assistant", "content": assistant_msg})
  # display
  for msg in st.session_state.history:
      role = "🙂" if msg["role"]=="user" else "🤖"
      st.markdown(f"**{role}:** {msg['content']}")
  ```

  The above stores conversation in session\_state, calls some `get_chat_response` that uses the conversation context, then displays all messages.

**Setup & Deployment:**

* Develop locally with `streamlit run app.py`. If pushing to GitHub, you can use **Streamlit Community Cloud** (streamlit.io) to deploy for free (with some resource limits). Or use Replit or Vercel via `streamlit` CLI (though Vercel isn’t ideal for persistent processes, but you could use Docker). Easiest: use Streamlit Share (the community cloud) by logging in with GitHub and sharing your repo, then you get a `yourname-yourrepo.streamlit.app` URL. This is great for hack demos as it’s one-click. But note their usage limits (e.g., limited CPU and 1GB RAM, plus a short idle shutdown time). For a single-user demo, it’s fine.
* If using Streamlit Cloud, ensure requirements.txt is in repo with needed packages (openai, etc.). The service will read that and install.
* One trick: if you need to hide API keys on Streamlit Cloud, you can add them as Secrets in the Streamlit cloud web UI (they’ll be in environment variables during runtime). Locally use .env file and load with python-dotenv.
* In Replit, running Streamlit may need a replit.nix config (since it’s not a simple one-file script, but essentially you can do `streamlit run app.py` in the Repl). Actually, Replit has a template for Streamlit that takes care of it. On Replit, you might just treat it as a normal Python app that listens on a port. Streamlit by default binds to localhost; on Replit, maybe specify `--server.enableCORS false --server.port $PORT --server.address 0.0.0.0` to run properly.

**Common Pitfalls:**

* *App re-runs on every widget change:* Streamlit’s paradigm triggers a re-run of the script with updated widget values each time there’s an interaction (except if you use session\_state). This can lead to some weird behavior if not accounted for, e.g., if your code above the button execution does heavy work, it might repeat. Best practice: put long processing inside the if button or so, not in the top-level that runs every time.
* *Blocking calls freeze UI:* Streamlit executes the script in a single thread. If you make a long API call, the app doesn’t respond until done. This is okay usually. But if you want to indicate progress, you can use `st.progress` or the placeholder trick to show a spinner or partial results. There’s no true multi-threading or async in Streamlit (there is experimental async support for some use cases, but not widely used). So sometimes you have to just accept the wait or break tasks into smaller chunks and use `st.sleep` to yield (though that still blocks the single thread). For hack demos, just calling API and waiting is fine.
* *Large outputs:* If you print a huge text blob, the app might become slow or crash. If you have extremely long LLM outputs or numerous messages, consider truncating or using `st.expander` to collapse it. Also, if you have to display code or JSON, use `st.code` which will give a scroll box.
* *Resource limits in cloud:* If you do something memory heavy (like load a big model locally in Streamlit Cloud – not recommended, their free tier doesn’t have GPU for sure and limited CPU), it might not handle it. Stick to API calls. If you absolutely need local models, consider using a smaller one and maybe splitting into separate processes (advanced). But likely out of scope for Buildathon – you’ll rely on OpenAI or similar.
* *Session separation:* Each user connecting gets their own session\_state by default – which is good (isolated). But if you want a multi-user chat where each sees each others messages, that’s complicated and not typical for hack; Streamlit isn’t multi-user interactive except each user sees their own session. It’s okay because typically hack demos are one user at a time.

**Why Streamlit for demo:** It gives a polished interface (with minimal design effort) and can incorporate charts, images, etc., easily to visualize results. It’s much faster to iterate than writing a Flask app + custom HTML/CSS. The downside is less control on layout specifics, but the default is usually fine (vertical scrolling app). It’s perfect for showing analysis steps, or an interactive playground for your LLM (like adjusting parameters live if you add sliders for temperature etc., since `st.slider` can adjust a value that you use in openai call). Judges often appreciate a clear UI demonstrating capabilities rather than just CLI or a notebook. Streamlit hits that sweet spot of ease and presentability.

<br>

## 14. Python + TypeScript Rapid Dev Stack

Using **Python and TypeScript together** can speed up development by leveraging Python’s quick prototyping (especially for ML tasks) and TypeScript’s strong front-end and serverless capabilities. A common pattern is: Python for data/ML, TS (Node or Next.js) for UI and integration. In Buildathon context, it might mean you build your core logic in Python (like a Flask API or a Jupyter prototype) and a front-end in TS (Next.js or plain React). Or perhaps use Node scripts for deployment and Python for experimentation.

**Tips for combining effectively:**

* **Define Clear Boundaries:** Decide which parts of your project each language handles. For instance:

  * *Python:* model inference, complex computations, database interactions (if using ORMs or ML libs). Possibly as a microservice or scheduled tasks.
  * *TypeScript:* front-end UI (web or mobile via React Native) and backend glue (serverless functions, API endpoints calling the Python service or directly calling third-party APIs, etc.). Also good for structured tasks and leveraging JS ecosystem (like a Chrome extension or such).
    By splitting, team members can work in parallel – Python person fine-tunes prompts or model usage, TS person designs UI/UX.

* **Communication between Python and TS:** If you make a Python backend (Flask/FastAPI/Django), your TS front-end can call it via HTTP (REST or GraphQL). E.g., Next.js `getServerSideProps` or API routes could fetch from the Python service. Alternatively, if using a Node backend (like an Express or Next.js API route) you can call Python via command line or RPC. For example:

  * Call a Python script from Node using `child_process.spawn` – handy for quick integration but consider overhead and environment issues.
  * Use a message queue or database as intermediary (a bit heavy for hackathon).
  * Simpler: host Python logic as an HTTP service (FastAPI makes it easy) and just fetch it. Or if using Replit or some PaaS, you could run both in same host on different ports (but complexity).
    For quick hack, often easiest is to keep things in one environment: e.g., if using Streamlit (Python) as UI, you might not need TS at all. Or if using Next.js as UI, try to implement logic there calling APIs. But if Python is non-negotiable (for a certain library), you can embed it. There are projects like `Pyodide` (Python in WebAssembly to run in browser) – not trivial but possible if you want to run some light ML in the browser. But usually, just call a Python server.

* **Rapid Dev Workflow:**

  * Use **hot-reloading** wherever possible: Next.js gives hot reload for UI, Flask with `debug=True` reloads on code changes, etc. This shortens iteration cycle.
  * For TS, leverage type definitions to catch errors early as you prototype. E.g., define interfaces for the data returned by Python API so you get autocompletion and don't mis-use it.
  * For Python, use notebooks to quickly test logic, then port to a script for integration. (But careful not to spend too much time duplicating – ideally plan out functions to move into script from the start).
  * Write small end-to-end tests as you go. e.g., from TS, call the Python API with a sample input to verify integration. Or vice versa use Python `requests` to hit the TS API route if it does part of logic.

* **Full-Stack Observability:** Logging and debugging across two languages can get tricky. Use consistent logging (maybe both log to console or a shared logging service). For example, when Python receives a request from TS, log an ID and have TS log the same ID when making the request. This way you can correlate in logs. If using Sentry or similar, you can integrate both sides for error tracking. But simplest: print statements with timestamps that you can follow manually.

* **Sharing Code/Models:** If you have some data structures or constants, avoid duplication by generating once and using in both, if possible. TypeScript cannot directly import Python code, but you can use JSON schema or .env files to share config. For instance, if you have a list of prompt templates or questions, maintain them in one place (maybe Python) and have TS fetch it or include it at build time (like reading a JSON output of Python script). This isn't always necessary, but for things like vocabulary lists or config values, it prevents mismatches.
  Another example: if you fine-tune a model in Python and output a `.onnx` or `.pkl`, the TS side might need to load it or at least know the label mapping etc. So output an artifact (file) from Python that TS can consume.

* **Example Setup:** Consider a scenario – you have a FastAPI app with an endpoint `/answer` that takes question and returns answer using some Python logic with LangChain and supabase. Then you have a Next.js front-end that uses `fetch('/answer?q=...')`. Steps:

  1. Code and run FastAPI locally on port 8000 (with auto-reload).
  2. In Next.js dev, set up a proxy or directly call `http://localhost:8000/answer` from `getServerSideProps` or an API route to avoid CORS in development. Or you could disable CORS in FastAPI during dev.
  3. Both the Python and Next dev servers run concurrently. As you tweak one or the other, refresh and test quickly.
  4. Deploy: perhaps deploy FastAPI to a service (maybe Deta Space or Heroku or a small VM) and Next.js to Vercel. Or for simplicity, deploy both in one container using something like `fly.io` that can run a Dockerfile with both (but mixing Python + Node in one container can increase complexity). Another approach is to use Next.js API routes for everything: you can run Python within those via spawn as mentioned. E.g., `pages/api/ask.js` uses `spawn('python', ['ai_script.py', query])` and captures stdout. This avoids hosting two servers, at the cost of some overhead on each call. For moderate usage it's fine. You must have the Python files accessible to Node (so include them in the Next project). And ensure any required pip libs are installed on the system. If deploying on Vercel, Vercel supports Python in Serverless Functions by detecting if you have a requirements.txt for those functions. But it's easier to just call an external API to be honest.

**Common Pitfalls:**

* *Dependency Hell:* Ensure the environments are resolved: e.g., if using spawn, the host must have Python and the required packages installed. On Vercel serverless, you might not have some large ML libs available (they restrict). Check if allowed. Possibly use a lightweight approach (maybe requests and minimal logic in that script). If heavy, better have a dedicated Python service.
* *CORS issues:* If front-end and back-end are separate, handle CORS. E.g., enable `FastAPI(cors_allowed_origins=['*'])` or configure allowed domain. During dev, might use a proxy or simply run front-end in same domain by hosting a static file that calls your service (not possible if different languages, so CORS is the way).
* *Time Sync for debugging:* Node uses MS for timestamps, Python datetime default is in s or logs in human times, unify these when comparing logs. Possibly use a shared format (like both log in ISO UTC). Not critical but can ease debugging concurrency issues.
* *Double Effort vs single:* Sometimes doing everything in one stack might be simpler – e.g., if the whole thing can be done in Python with Streamlit, you avoid TS. Or if Node has all needed libs (these days Node also has LangChain, Pinecone, etc.). But often Python ML libs are richer, so we use that, and TS for UI. Accept some overhead in integration but focus on dividing tasks so each side’s strengths shine.

**Rapid Dev Stack Summary:** Use Python for what it's best at (ML, quick scripting) and TS/JS for UI and robust web handling. They can be connected through simple HTTP calls, and this separation can make development parallel and play to each developer’s expertise. Just keep the interface between them as simple and well-defined as possible (e.g., a REST API with well-defined JSON inputs/outputs) to reduce integration bugs.

<br>

## 15. Prompt Engineering & RAG Patterns

Finally, a crucial skill in any LLM project: **Prompt Engineering** – crafting effective prompts to get desired outputs – and **RAG (Retrieval-Augmented Generation)** patterns – combining prompts with retrieved context to reduce hallucinations and augment knowledge.

**Cheat Sheet – Prompt Engineering Best Practices:**

* **Clarity and Specificity:** Clearly instruct the model. If you want a certain format, say it explicitly. E.g., “Provide the answer in JSON format with keys 'analysis' and 'recommendation'.” Use step-by-step language if needed (“First, analyze the input. Then, output a recommendation.”). Models like instructions spelled out.
* **Role Prompting:** Setting context via system prompt or initial text: e.g., “You are an expert travel guide.” This can influence tone and knowledge depth. Use it to establish the persona or expertise level needed. However, don’t overly constrain unless necessary (some creativity can be lost if too rigid).
* **Chain-of-Thought (CoT):** Encourage the model to reason step by step by either asking it to “show your reasoning” (and then maybe parse out final answer) or use hidden CoT (some frameworks use a prompt that asks model to think in an intermediate step). In open-ended QA, a simple trick: “Think step-by-step about the problem before giving final answer.” often yields better results. If using GPT-4 with function calling, you might not need visible CoT, but for earlier models it helps to break tasks down.
* **Examples (Few-shot):** Provide example question-answer pairs to shape responses (especially for GPT-3.5 and older). For instance, if building a translator, show an example input and output. Few-shot helps the model infer format and style. With the context window available now (up to 4k or more), you can include 2-5 examples if it significantly improves performance. Balance: too many examples can reduce space for user input or lead model to copy them. Choose representative ones.
* **Avoid Open-Ended Triggers if not needed:** The model often tries to be verbose. If you need a concise answer, say “Answer with at most one sentence.” or “Give a short answer.” Otherwise asking “Tell me about X” yields a paragraph usually.
* **Use of System vs User vs Assistant roles:** In OpenAI’s chat API, the **system** message should contain the primary directives (e.g., style, persona, constraints), user message contains the question, and assistant messages might contain prior context or examples. For Anthropic’s Claude, just providing a single prompt with an “assistant:” role or using their simpler prompt format can suffice, but they also support system (they call it “Claude’s Constitution” in some writeups). In function-calling, system can define functions and behavior, user asks, model can produce a function call.
* **Be Mindful of Bias and Content:** If the user might input something that triggers content filters (violence, etc.), your prompt could pre-handle it: e.g., “If the user asks something disallowed, respond with a brief apology and refusal.” This sets expectation and can avoid model getting itself into trouble. However, the base models have their own safety layers too. Just ensure your prompt doesn’t accidentally encourage disallowed content (e.g., don’t say “You can answer anything freely without restrictions” unless you want it to violate the content rules – which it likely won’t but might lead to unpredictable results or refusals).
* **Iterative Prompt Refinement:** Try variations of your prompt and evaluate outputs. Minor wording changes can matter. For example, adding “You are a world-class poet” before a prompt can drastically change style. Use that to your advantage: if style is wrong, adjust role or add “Use a casual tone” etc. If info is missing, maybe instruct “If the answer is not in context, say you don’t know (don’t hallucinate).”

**RAG (Retrieval-Augmented Generation) Patterns:**
This approach is about augmenting the prompt with relevant retrieved data, usually from a vector store, to help the model answer with factual correctness. Key steps in RAG:

1. **Embed Query & Retrieve:** Take user question, generate embedding, find top-k relevant docs (as shown with Supabase/Mongo vector search earlier).
2. **Construct Prompt with Context:** Insert those retrieved texts into the prompt, often as a context section or as part of system message. Common template:

   ```
   Use the following context to answer the question.
   Context:
   {{doc1}}
   {{doc2}}
   Question: {{user question}}
   Answer in a concise manner:
   ```

   The context might be large, so be mindful of token limits – often use top 3-5 snippets. Sometimes adding bullet points or a format to context can help model parse it better, but generally raw text is okay.
3. **Ask model to cite or not?** If needed, you can instruct the model to mention sources (like “Include the source name in parentheses after each fact you use.”). This works if context snippets have an identifier, e.g., “\[Doc1] ...text...”. The model then could refer “\[Doc1]”. GPT-4 does this fairly well if asked. This adds credibility. However, it might break fluency or if not needed, skip it.
4. **Handle If Answer Not Found:** If context doesn’t contain answer, the model might guess or make up. To mitigate: explicitly say “If the answer is not in the context, respond 'I don't know'.” Models will follow that instruction somewhat, especially GPT-4. But they might still guess if context is tangentially related. Another tactic: after getting answer, do a quick check – e.g., require that any answer has some overlap with context. If not, either refuse or run a second retrieval with a broader query. This is more advanced (LangChain has such refine steps). For hack, at least instruct it not to use outside knowledge.
5. **Iteration between Retrieval and Generation:** Sometimes one round of retrieval isn’t enough (if question is broad). You can either retrieve more chunks or do a multi-turn approach: ask model to come up with search terms if initial attempt fails. This enters **Conversational Retrieval QA** territory – e.g., if the model says “I need more info about X”, your system could capture that and do another vector query. Tools like LlamaIndex or LangChain can automate such loops. In a short buildathon, you may just pick a good `k` and hope for best. But keep the possibility in mind if quality issues arise – often increasing `k` from 3 to 5 or 7 can help get all needed info, at cost of more tokens.

**Example Prompt with RAG:**
Suppose building a medical Q\&A with a knowledge base. A template might be:

```plaintext
You are a helpful medical assistant. Answer questions based only on the following documents. 
If you don't find the answer in them, say "I'm not sure".

DOCUMENTS:
Doc1: In diabetes, the body cannot regulate blood sugar due to lack of insulin...
Doc2: Insulin is a hormone produced by the pancreas that lowers blood glucose...
Doc3: [more content]

QUESTION: How does insulin affect blood sugar?
```

And then let the model answer. The model should integrate info from Doc1 and Doc2 to answer. The prompt explicitly forbids outside info and tells what to do if not found. This greatly reduces hallucination because it has high-quality info to draw from.

**Common Pitfalls in RAG:**

* *Too much or Irrelevant Context:* If you stuff a lot of documents, model might get distracted or use wrong piece. It can also cause the answer to be overly verbose summarizing them. It’s often better to pass just the most relevant chunks. Ensure your retrieval actually returns relevant things (vector search sometimes can return things that aren't obviously related due to semantic weirdness). You might add an intermediate check: e.g., filter out any snippet that has very low similarity score or that doesn’t at least mention some keywords from the question. Another trick: prefix each chunk with a title or source name so model knows which is which.
* *Model still hallucinating:* Even with context, models sometimes mix context with prior knowledge incorrectly. Reiterate in prompt to use given docs only. If it still does, maybe the context wasn’t clear or it trusts its memory more (especially if context is smaller or lower quality than model’s training on that topic). GPT-4 is pretty good at using provided context if told. GPT-3.5 is hit or miss – it might ignore instructions more. If using 3.5, you might find it injecting outside facts. To mitigate, you can try to detect references not present in context (hard programmatically). Another approach: include a dummy irrelevant text in context and instruct the model to ignore unrelated info – this tests if it can differentiate. But ideally, use a strong model or very explicit instructions.
* *Token limits:* Always watch total tokens (user prompt + context + model answer). If using GPT-3.5 (4k) and you put 3500 tokens of docs, the answer might be cut off or it might refuse. For GPT-4 (8k+), more breathing room but still, try not to fill beyond \~70% with context. Summarize or chunk docs if needed.
* *Latency:* Retrieving from DB + calling LLM = two network calls possibly, can slow responses. Use caching if possible (if same query repeats, cache the answer or at least the embedding+results). Helicone (Section 7) can help measure how long each part takes. Perhaps embed and store common question embeddings in a dictionary to skip vector search for those (if you know a set of likely Qs, but not usually in open Q\&A).

Combining prompt engineering and RAG effectively: The better your prompt framing and the quality of retrieved context, the better your results. It transforms the LLM from a general model to a specific domain expert that “cites” a knowledge base. This is the backbone of many production LLM applications now.

---

## Comparison of Agentic Frameworks (LangGraph vs CrewAI vs AutoGen vs OpenAI SDK)

To wrap up, here’s a quick comparison matrix of the agent frameworks we discussed, highlighting their strengths and trade-offs:

| Framework               | Language & Scope         | Strengths                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Weaknesses / Trade-offs                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **LangChain LangGraph** | Python (LangChain ext.)  | - Seamless integration with LangChain ecosystem (chains, tools, memory).<br>- Supports **cyclic graphs** and multi-agent loops (complex workflows).<br>- Good for **production debugging** with LangSmith integration (tracing each step).                                                                                                                                                                                                                           | - Steeper learning curve due to many abstractions.<br>- Tightly coupled to LangChain versions (upgrades can break things).<br>- Overhead of LangChain – may be heavy for simple tasks.                                                                                                                                                                                                                                                           |
| **CrewAI**              | Python (independent)     | - Built **from scratch** for multi-agent autonomy – very performant (5.7× faster claims vs LangChain).<br>- High-level Crews for agent teams, and Flows for deterministic logic.<br>- Large library of tools and strong community (100k+ devs).<br>- Designed for enterprise: observability, testing, YAML config for quick setup.                                                                                                                                   | - Slightly **opinionated structure** – you must frame problem as crew of agents and tasks, which might be overkill for simple cases.<br>- Newer framework – documentation is evolving, some rough edges (e.g., dependency on UV tool management might confuse some users).<br>- Fewer third-party examples (compared to LangChain's multitude).                                                                                                  |
| **Microsoft Autogen**   | Python (open-source)     | - Focused on *conversation between agents* – easy to set up multi-agent chat loops.<br>- Good for scenarios like code assistant (with one agent writing code and another executing or critiquing).<br>- Backed by MSR – comes with examples and presumably some support.<br>- Integrates code execution seamlessly (UserProxyAgent auto-runs code blocks).                                                                                                           | - Conversation-centric: less built-in notion of a directed process (no explicit concept of tasks/goals beyond what you prompt).<br>- Beta state (knnBeta naming suggests not final) – API changes possible.<br>- Lacks a big community or rich tooling around it (no dedicated UI for tracing like LangSmith, though you can print logs).                                                                                                        |
| **OpenAI Agents SDK**   | Python (with OpenAI API) | - **Lightweight & minimalistic** – only a few primitives, quick to learn.<br>- **Built-in tool/function support** – turning Python funcs to tools easily, uses OpenAI function calling under the hood.<br>- **Session management** and guardrails baked in – easier to handle multi-turn with memory and input validation.<br>- Comes with visualization/tracing (OpenAI had a UI for agent runs, possibly integrated with dev console or via Temporal integration). | - Tied to OpenAI’s ecosystem – for example, uses their function calling (works great with OpenAI models, less with others unless through compatibility layer).<br>- Still new (released mid-2023) – might have some maturity issues or fewer community examples.<br>- Fewer abstractions means you might need to implement some patterns yourself (like custom multi-agent handoff logic beyond provided sample) – though simpler is often fine. |

**Tool Selection Trade-offs:** When picking frameworks or tools:

* If your project heavily uses **LangChain already (for vector DB, memory, etc.)**, LangGraph is logical – it extends your existing setup. If performance issues arise, consider CrewAI as a faster alternative (CrewAI even provides a comparison showing LangGraph vs CrewAI with CrewAI being faster and more flexible in some ways).
* For a quick **multi-agent proof-of-concept**, OpenAI’s SDK might be the fastest route (especially if you primarily need one agent with tools, or a straightforward handoff scenario). It handles a lot out of the box (tool calling, looping until done) with minimal code.
* If you need **complex agent collaboration** (like multiple specialists solving a big task, with a manager agent coordinating), CrewAI is designed for that scenario out-of-the-box. It might save you time by providing the structure (roles, tasks, flows) so you don’t have to code that coordination logic from scratch.
* If you specifically are doing a **chatbot with self-reflection or code generation involving execution**, AutoGen’s pattern of Assistant + UserProxy (with code exec) is very powerful. It could solve code questions by writing and running code all within the chat loop, which is hard to replicate exactly in others without custom coding.

In practice, you might mix and match: e.g., use OpenAI SDK for a main agent with tools, but incorporate a CrewAI flow if you later scale up to multiple agents. But given hackathon time, it's usually best to pick one and stick to it to avoid integration overhead.

**Deploying & Demo Tips:** Finally, after building all these components, showing a polished demo fast involves:

* **Live logs & observability:** as discussed, use Helicone or LangSmith so you can monitor usage during the demo (and if something goes wrong, you have insight). Also, showing a live dashboard of usage (requests count, etc.) can impress judges that you thought about monitoring. But do this only if time permits and it adds value to explain.
* **Latency tricks:** as mentioned in section 7, streaming outputs keep users engaged even if total completion takes long – they see progress. Also, consider concurrency: if your app can parallelize tasks (like fetch multiple docs at once or call two different APIs simultaneously), do it. For example, if you need embeddings from two different models, call them async concurrently. Also for long-running tasks, maybe prepare a cached result to show quickly, then explain more after if needed. For demo, perceivable speed often matters more than actual – so using a smaller model for demo could be wise if it’s much faster and the answer quality is acceptable.
* **UI polish:** if possible, add small touches like a spinner or a “Generating answer...” message while waiting for response, rather than freezing. In Streamlit, you could do `with st.spinner("Thinking..."):` context. In Next.js, you might show a loading state. Little things show you care about UX.
* **Backup plan:** have a fallback if the AI fails. E.g., if OpenAI API is down or your key quota exhausted right before demo, have either a pre-computed answer or a local model as backup. Even a simple rule-based answer “Sorry, I cannot answer now.” is better than a crash. Judges understand if an API hiccups, but a graceful handling is appreciated.
* **Time management in demo:** likely you have a few minutes to present. Focus on the features you built (graph connections, agent autonomy, etc.) rather than going too deep into code. A high-level architecture diagram can help (e.g., show how LLM, vector DB, and UI interact). Then show the actual app solving a sample problem. Use a test case that you know works well (maybe not the hardest edge-case, unless your selling point is handling edge cases).
* **Emphasize trade-off decisions:** If asked why you picked a certain tech (like “Why Supabase over Pinecone?” or “Why CrewAI over LangChain?”), have an answer: maybe ease of setup, familiarity, open-source nature, etc. Showing you considered alternatives (maybe in Q\&A, mention “We also evaluated X but chose Y because...”) can score points for engineering thoughtfulness.

Good luck building and demonstrating your **graph-powered, LLM-fueled, fast-stack** project! With the comprehensive knowledge in this textbook, you have all the reference needed to win that Buildathon 🏆.
