from database import insert_decision_method

# Existing Methods
insert_decision_method(
    "Fuzzy AHP",
    "A fuzzy-based analytic hierarchy process for ranking decision criteria in uncertain environments.",
    "Saaty (2005)"
)

insert_decision_method(
    "VIKOR Method",
    "A multi-criteria optimization method that finds compromise solutions between conflicting objectives.",
    "Opricovic & Tzeng (1998)"
)

# 🌍 Additional Decision-Making Methods from td-net Toolbox

insert_decision_method(
    "Multi-Criteria Decision Analysis (MCDA)",
    "Evaluates multiple conflicting decision criteria using structured approaches.",
    "Belton & Stewart (2002)"
)

insert_decision_method(
    "Grey Relational Analysis (GRA)",
    "Analyzes relationships between factors in complex decision scenarios.",
    "Deng (1989)"
)

insert_decision_method(
    "Ordinal Priority Approach (OPA)",
    "Ranks alternatives based on ordinal preferences with fuzzy adaptation.",
    "Ahmadi, Shaban & Zavadskas (2021)"
)

insert_decision_method(
    "Delphi Method",
    "Uses expert consensus through iterative surveys to guide decision-making.",
    "Dalkey & Helmer (1963)"
)

insert_decision_method(
    "Scenario Planning",
    "Explores future uncertainties by constructing plausible scenarios.",
    "Schoemaker (1995)"
)

insert_decision_method(
    "System Dynamics Modeling",
    "Simulates complex system behaviors using feedback loops and stock-flow structures.",
    "Forrester (1961)"
)

insert_decision_method(
    "Soft Systems Methodology (SSM)",
    "Uses qualitative analysis to structure ill-defined problems.",
    "Checkland (1981)"
)

insert_decision_method(
    "Fuzzy Cognitive Maps (FCMs)",
    "Models causal relationships between factors in uncertain environments.",
    "Kosko (1986)"
)

insert_decision_method(
    "Social Network Analysis (SNA)",
    "Examines the impact of network structures on decision-making.",
    "Wasserman & Faust (1994)"
)

print("✅ Successfully inserted additional decision-making methods!")
