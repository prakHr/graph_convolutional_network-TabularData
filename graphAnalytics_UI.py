# ============================================================
# DASH APP
# MANUAL SENTENCE INPUT
#        ↓
# SENTENCE LIST
#        ↓
# BERTopic
#        ↓
# MandalaGraph
#        ↓
# GRAPH ANALYTICS
# ============================================================

from transformers.utils import logging
from huggingface_hub import utils

# ------------------------------------------------------------
# Disable model-loading progress bars
# ------------------------------------------------------------

logging.disable_progress_bar()
utils.disable_progress_bars()


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

import os

os.environ["OMP_NUM_THREADS"] = "1"


# ------------------------------------------------------------
# Standard libraries
# ------------------------------------------------------------

import multiprocessing

from pprint import pprint


# ------------------------------------------------------------
# Third-party libraries
# ------------------------------------------------------------

import pandas as pd

from dash import (
    Dash,
    html,
    dcc,
    Input,
    Output,
    State,
    no_update
)

import dash_bootstrap_components as dbc

from mpire import WorkerPool

from bertopic import BERTopic

from mandala_gnn import (
    MandalaGraph,
    CentralityAnalyzer,
    CommunityDetector,
    GraphMetrics
)


# ============================================================
# 1. CREATE SCALABLE EDGES AND NODES
# ============================================================

def create_scalable_edges_and_nodes(list_of_topic):

    edges = []
    nodes = []

    # --------------------------------------------------------
    # BERTopic Representation
    #
    # Example:
    #
    # ['learning', 'machine', 'like', '', '', ...]
    # --------------------------------------------------------

    if isinstance(list_of_topic, list):

        topic_words = [
            str(word).strip()
            for word in list_of_topic
            if str(word).strip()
        ]

    else:

        topic_words = [
            str(list_of_topic).strip()
        ]

    # Remove duplicates while preserving order
    topic_words = list(
        dict.fromkeys(topic_words)
    )

    # --------------------------------------------------------
    # Add nodes
    # --------------------------------------------------------

    nodes.extend(topic_words)

    # --------------------------------------------------------
    # Connect words inside the topic
    #
    # Example:
    #
    # [learning, machine, like]
    #
    # produces:
    #
    # learning -> machine
    # learning -> like
    # machine  -> like
    # --------------------------------------------------------

    for i in range(
        len(topic_words)
    ):

        for j in range(
            i + 1,
            len(topic_words)
        ):

            edges.append(
                (
                    topic_words[i],
                    topic_words[j]
                )
            )

    return {
        "edges": edges,
        "nodes": nodes
    }
# ============================================================
# 2. CREATE KNOWLEDGE GRAPH
# ============================================================

def create_knowledge_graph(
    list_of_sentences,
    topic_name_column="Representation"
):

  
    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not list_of_sentences:

        raise ValueError(
            "No sentences were supplied."
        )

    # --------------------------------------------------------
    # Clean sentences
    # --------------------------------------------------------

    docs = [
        str(sentence).strip()
        for sentence in list_of_sentences
        if str(sentence).strip()
    ]

    if not docs:

        raise ValueError(
            "No valid sentences were supplied."
        )

    
    # --------------------------------------------------------
    # BERTopic needs multiple documents.
    #
    # If only a few sentences are entered, replicate them
    # internally.
    #
    # This does NOT modify the user's sentence list.
    # --------------------------------------------------------

    model_docs = list(docs)

    if len(model_docs) < 10:

        multiplier = (
            10 // len(model_docs)
        ) + 1

        model_docs = (
            model_docs * multiplier
        )[:10]


    # --------------------------------------------------------
    # BERTopic
    # --------------------------------------------------------

    topic_model = BERTopic(
        verbose=False
    )

    topics, probs = topic_model.fit_transform(
        model_docs
    )

    # --------------------------------------------------------
    # Topic information
    # --------------------------------------------------------

    topic_df = topic_model.get_topic_info()

    
    # --------------------------------------------------------
    # Check Representation column
    # --------------------------------------------------------

    if topic_name_column not in topic_df.columns:

        raise ValueError(
            f"Column '{topic_name_column}' "
            f"not found in BERTopic output."
        )

    # --------------------------------------------------------
    # Get topic representations
    # --------------------------------------------------------

    list_of_topics = (
        topic_df[
            topic_name_column
        ]
        .tolist()
    )

    list_of_topics = [
        topic
        for topic in list_of_topics
        if topic is not None
    ]

    # --------------------------------------------------------
    # Prepare MPire input
    # --------------------------------------------------------

    results = []

    for topic in list_of_topics:

        results.append(
            {
                "list_of_topic": topic
            }
        )

    # --------------------------------------------------------
    # MPire
    # --------------------------------------------------------

    num_cores = max(
        multiprocessing.cpu_count() // 2,
        1
    )

    
    with WorkerPool(
        n_jobs=num_cores,
        daemon=False
    ) as pool:

        results = pool.map(
            create_scalable_edges_and_nodes,
            results,
            progress_bar=False
        )

    # --------------------------------------------------------
    # Create MandalaGraph
    # --------------------------------------------------------

    graph = MandalaGraph()

    edges_list = []
    nodes_list = []

    for result in results:

        edges_list.extend(
            result["edges"]
        )

        nodes_list.extend(
            result["nodes"]
        )

    # --------------------------------------------------------
    # Remove duplicate nodes
    # --------------------------------------------------------

    unique_nodes = []

    for node in nodes_list:

        if node not in unique_nodes:

            unique_nodes.append(node)

    # --------------------------------------------------------
    # Add nodes
    # --------------------------------------------------------

    for node in unique_nodes:

        graph.add_node(node)

    # --------------------------------------------------------
    # Add edges
    # --------------------------------------------------------

    for edge in edges_list:

        graph.add_edge(
            edge[0],
            edge[1]
        )

    
    return graph


# ============================================================
# 3. GRAPH ANALYSIS
# ============================================================

def graph_analysis(graph):

    # --------------------------------------------------------
    # Centrality
    # --------------------------------------------------------

    analyzer = CentralityAnalyzer(
        graph
    )

    # --------------------------------------------------------
    # Community detection
    # --------------------------------------------------------

    detector = CommunityDetector(
        graph
    )

    communities = detector.louvain()

    # --------------------------------------------------------
    # Graph metrics
    # --------------------------------------------------------

    metrics = GraphMetrics(
        graph
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {

        "pagerank":
            analyzer.pagerank(),

        "betweenness":
            analyzer.betweenness(),

        "communities":
            communities,

        "modularity":
            detector.modularity(
                communities
            ),

        "graph_metrics":
            metrics.summary()
    }

    return results


# ============================================================
# 4. CONVERT DATA TO DATAFRAME
# ============================================================

def make_dataframe(
    data,
    first_column="Key"
):

    if data is None:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(data, dict):

        rows = []

        for key, value in data.items():

            rows.append(
                {
                    first_column:
                        str(key),

                    "Value":
                        str(value)
                }
            )

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # List / Tuple
    # --------------------------------------------------------

    if isinstance(
        data,
        (list, tuple)
    ):

        rows = []

        for index, value in enumerate(
            data
        ):

            rows.append(
                {
                    "Index":
                        index,

                    first_column:
                        str(value)
                }
            )

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # Single value
    # --------------------------------------------------------

    return pd.DataFrame(
        [
            {
                first_column:
                    str(data)
            }
        ]
    )


# ============================================================
# 5. DATAFRAME → DASH TABLE
# ============================================================

def dataframe_to_table(df):

    if df is None or df.empty:

        return html.Div(
            "No data available.",
            className="text-muted"
        )

    return dbc.Table.from_dataframe(
        df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="mt-3"
    )


# ============================================================
# 6. CREATE ANALYTICS UI
# ============================================================

def create_analytics_ui(
    results,
    sentences
):

    # --------------------------------------------------------
    # Convert analytics to DataFrames
    # --------------------------------------------------------

    pagerank_df = make_dataframe(
        results["pagerank"],
        "Node"
    )

    betweenness_df = make_dataframe(
        results["betweenness"],
        "Node"
    )

    communities_df = make_dataframe(
        results["communities"],
        "Community"
    )

    graph_metrics_df = make_dataframe(
        results["graph_metrics"],
        "Metric"
    )

    # --------------------------------------------------------
    # Modularity
    # --------------------------------------------------------

    modularity = results[
        "modularity"
    ]

    if isinstance(
        modularity,
        (float, int)
    ):

        modularity_display = (
            f"{modularity:.4f}"
        )

    else:

        modularity_display = str(
            modularity
        )

    # ========================================================
    # ANALYTICS UI
    # ========================================================

    return [

        html.H2(
            "Analytics",
            className="mb-4"
        ),

        # ----------------------------------------------------
        # SUMMARY CARDS
        # ----------------------------------------------------

        dbc.Row(
            [

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [

                                html.H6(
                                    "Sentences"
                                ),

                                html.H2(
                                    len(sentences)
                                )

                            ]
                        )
                    ),
                    width=3
                ),

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [

                                html.H6(
                                    "Modularity"
                                ),

                                html.H2(
                                    modularity_display
                                )

                            ]
                        )
                    ),
                    width=3
                ),

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [

                                html.H6(
                                    "Communities"
                                ),

                                html.H2(
                                    len(
                                        communities_df
                                    )
                                )

                            ]
                        )
                    ),
                    width=3
                ),

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [

                                html.H6(
                                    "Graph Metrics"
                                ),

                                html.H2(
                                    len(
                                        graph_metrics_df
                                    )
                                )

                            ]
                        )
                    ),
                    width=3
                )

            ],
            className="mb-4"
        ),

        # ----------------------------------------------------
        # PAGERANK
        # ----------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H4(
                        "PageRank"
                    ),

                    html.P(
                        (
                            "Importance of nodes "
                            "within the graph."
                        ),
                        className="text-muted"
                    ),

                    dataframe_to_table(
                        pagerank_df
                    )

                ]
            ),
            className="mb-4"
        ),

        # ----------------------------------------------------
        # BETWEENNESS
        # ----------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H4(
                        "Betweenness Centrality"
                    ),

                    html.P(
                        (
                            "Nodes that act as "
                            "important bridges "
                            "in the graph."
                        ),
                        className="text-muted"
                    ),

                    dataframe_to_table(
                        betweenness_df
                    )

                ]
            ),
            className="mb-4"
        ),

        # ----------------------------------------------------
        # COMMUNITIES
        # ----------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H4(
                        "Louvain Communities"
                    ),

                    html.P(
                        (
                            "Communities detected "
                            "from the knowledge graph."
                        ),
                        className="text-muted"
                    ),

                    dataframe_to_table(
                        communities_df
                    )

                ]
            ),
            className="mb-4"
        ),

        # ----------------------------------------------------
        # MODULARITY
        # ----------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H4(
                        "Modularity"
                    ),

                    html.H2(
                        modularity_display
                    )

                ]
            ),
            className="mb-4"
        ),

        # ----------------------------------------------------
        # GRAPH METRICS
        # ----------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H4(
                        "Graph Metrics"
                    ),

                    dataframe_to_table(
                        graph_metrics_df
                    )

                ]
            ),
            className="mb-4"
        )
    ]


# ============================================================
# 7. DASH APP
# ============================================================

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG
    ]
)

app.title = (
    "Sentence Knowledge Graph Analytics"
)


# ============================================================
# 8. DASH LAYOUT
# ============================================================

app.layout = dbc.Container(
    [

        # ====================================================
        # HEADER
        # ====================================================

        dbc.Row(
            dbc.Col(
                [

                    html.H1(
                        "Sentence Knowledge Graph",
                        className=(
                            "text-center "
                            "mt-4"
                        )
                    ),

                    html.P(
                        (
                            "Enter sentences manually, "
                            "build a sentence list, "
                            "and analyze the resulting "
                            "knowledge graph."
                        ),
                        className=(
                            "text-center "
                            "text-muted"
                        )
                    )

                ],
                width=12
            )
        ),

        html.Hr(),

        # ====================================================
        # SENTENCE INPUT
        # ====================================================

        dbc.Card(
            dbc.CardBody(
                [

                    html.H4(
                        "Enter Sentence"
                    ),

                    dbc.Textarea(
                        id="sentence-input",

                        placeholder=(
                            "Type your sentence here..."
                        ),

                        style={
                            "height": "120px"
                        },

                        className="mt-3"
                    ),

                    dbc.Button(
                        "Add Sentence",

                        id="add-sentence-btn",

                        color="primary",

                        size="lg",

                        className="mt-3"
                    )

                ]
            ),
            className="mb-4"
        ),

        # ====================================================
        # CURRENT SENTENCE
        # ====================================================

        dbc.Card(
            dbc.CardBody(
                [

                    html.H4(
                        "Current Sentence"
                    ),

                    html.Div(
                        id="current-sentence",

                        children=(
                            "No sentence added yet."
                        ),

                        className=(
                            "mt-3 fs-5"
                        )
                    )

                ]
            ),
            className="mb-4"
        ),

        # ====================================================
        # SENTENCE LIST
        # ====================================================

        dbc.Card(
            dbc.CardBody(
                [

                    html.H4(
                        "Sentence List"
                    ),

                    html.Div(
                        id="sentence-list",

                        children=(
                            html.P(
                                (
                                    "Your added "
                                    "sentences will "
                                    "appear here."
                                ),
                                className=(
                                    "text-muted"
                                )
                            )
                        ),

                        className="mt-3"
                    )

                ]
            ),
            className="mb-4"
        ),

        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        dbc.Row(
            dbc.Col(
                dbc.Button(
                    "Analyze Sentences",

                    id="analyze-btn",

                    color="success",

                    size="lg",

                    className="w-100"
                ),

                width=12
            ),

            className="mb-4"
        ),

        # ====================================================
        # ANALYTICS
        # ====================================================

        html.Div(
            id="analytics-section"
        ),

        # ====================================================
        # STORE
        # ====================================================

        dcc.Store(
            id="sentence-store",
            data=[]
        ),

        # ====================================================
        # STATUS
        # ====================================================

        html.Div(
            id="status",

            className=(
                "text-center "
                "mt-3 mb-4"
            )
        )

    ],

    fluid=True,

    className="px-5"
)


# ============================================================
# 9. ADD SENTENCE CALLBACK
# ============================================================

@app.callback(
    Output(
        "current-sentence",
        "children"
    ),

    Output(
        "sentence-list",
        "children"
    ),

    Output(
        "sentence-store",
        "data"
    ),

    Output(
        "sentence-input",
        "value"
    ),

    Input(
        "add-sentence-btn",
        "n_clicks"
    ),

    State(
        "sentence-input",
        "value"
    ),

    State(
        "sentence-store",
        "data"
    ),

    prevent_initial_call=True
)
def add_sentence_callback(
    n_clicks,
    sentence,
    sentences
):

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not sentence or not sentence.strip():

        return (
            dbc.Alert(
                (
                    "Please enter a sentence."
                ),
                color="warning"
            ),

            no_update,

            no_update,

            no_update
        )

    # --------------------------------------------------------
    # Get existing sentences
    # --------------------------------------------------------

    if sentences is None:

        sentences = []

    sentences = list(
        sentences
    )

    # --------------------------------------------------------
    # Clean sentence
    # --------------------------------------------------------

    sentence = sentence.strip()

    # --------------------------------------------------------
    # Add sentence
    # --------------------------------------------------------

    sentences.append(
        sentence
    )

    # --------------------------------------------------------
    # Build sentence list
    # --------------------------------------------------------

    sentence_items = []

    for index, value in enumerate(
        sentences,
        start=1
    ):

        sentence_items.append(
            dbc.ListGroupItem(
                [

                    html.Strong(
                        f"{index}. "
                    ),

                    value

                ]
            )
        )

    sentence_list_component = (
        dbc.ListGroup(
            sentence_items
        )
    )

    # --------------------------------------------------------
    # Return
    #
    # Empty input box after adding sentence.
    # --------------------------------------------------------

    return (

        sentence,

        sentence_list_component,

        sentences,

        ""

    )


# ============================================================
# 10. ANALYZE SENTENCES CALLBACK
# ============================================================

@app.callback(
    Output(
        "analytics-section",
        "children"
    ),

    Output(
        "status",
        "children"
    ),

    Input(
        "analyze-btn",
        "n_clicks"
    ),

    State(
        "sentence-store",
        "data"
    ),

    prevent_initial_call=True
)
def analyze_callback(
    n_clicks,
    sentences
):

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not sentences:

        return (

            dbc.Alert(
                (
                    "Please add at least "
                    "one sentence first."
                ),
                color="warning"
            ),

            "No sentences available."

        )

    # --------------------------------------------------------
    # Convert stored data into list of sentences
    # --------------------------------------------------------

    list_of_sentences = [

        str(sentence).strip()

        for sentence in sentences

        if str(sentence).strip()

    ]

   
    # --------------------------------------------------------
    # Run your original pipeline
    # --------------------------------------------------------

    try:

        # ----------------------------------------------------
        # Step 1: Create knowledge graph
        # ----------------------------------------------------

        graph = create_knowledge_graph(
            list_of_sentences,
            "Representation"
        )

        # ----------------------------------------------------
        # Step 2: Analyze graph
        # ----------------------------------------------------

        results = graph_analysis(
            graph
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        
        # ----------------------------------------------------
        # Create Dash analytics
        # ----------------------------------------------------

        analytics_ui = (
            create_analytics_ui(
                results,
                list_of_sentences
            )
        )

        return (

            analytics_ui,

            "Analytics generated successfully."

        )

    except Exception as e:

        # ----------------------------------------------------
        # Print error
        # ----------------------------------------------------


        return (

            dbc.Alert(
                [

                    html.H4(
                        "Analytics Error"
                    ),

                    html.P(
                        str(e)
                    )

                ],
                color="danger"
            ),

            "Analytics failed."

        )


# ============================================================
# 11. RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="localhost",
        port=8050
    )
