"""A small, fixed, offline sentence corpus spanning varied topics.

Used both to train the demo SAE and to power the "which other sentences
activate this feature" lookup in the Streamlit app. Kept small and
hardcoded on purpose so the demo has zero external data dependency
beyond the GPT-2 weights themselves.
"""

SENTENCES = [
    # technology
    "The server crashed after the deployment went out at midnight.",
    "She refactored the codebase to remove duplicate logic.",
    "The GPU ran out of memory during training.",
    "Kubernetes restarted the pod automatically after the crash.",
    "He wrote a script to automate the nightly backups.",
    "The API returned a 500 error under heavy load.",
    "They migrated the database to a managed cloud service.",
    "The neural network overfit the training data.",
    "A new firmware update fixed the bluetooth bug.",
    "The compiler flagged an unused variable warning.",
    # nature
    "The river froze solid during the long winter.",
    "A flock of geese flew south before the storm.",
    "The forest floor was covered in fallen leaves.",
    "Waves crashed against the rocky shoreline all night.",
    "The mountain peak was hidden behind thick clouds.",
    "A gentle breeze moved through the tall grass.",
    "The desert sand shifted with every gust of wind.",
    "Wildflowers bloomed across the entire hillside.",
    "The glacier has retreated significantly over the decades.",
    "A single eagle circled slowly above the valley.",
    # sports
    "The team celebrated wildly after the final whistle.",
    "She sprinted the last hundred meters to win the race.",
    "The coach called a timeout with two minutes left.",
    "He scored the winning goal in extra time.",
    "The crowd roared as the underdog took the lead.",
    "Injuries forced the star player to sit out the season.",
    "The referee reviewed the play on the replay screen.",
    "They trained for months before the championship game.",
    "The marathon runner collapsed just short of the finish line.",
    "A last-second three-pointer won the match.",
    # food
    "The bread was still warm from the oven.",
    "She added too much salt to the soup by accident.",
    "The restaurant was famous for its spicy noodles.",
    "He grilled vegetables over an open flame.",
    "The cake collapsed because the oven door was opened too early.",
    "Fresh basil made the sauce taste much brighter.",
    "They shared a bottle of wine over a long dinner.",
    "The coffee was bitter without any sugar.",
    "A street vendor sold roasted chestnuts on the corner.",
    "The soup simmered slowly for several hours.",
    # emotion / relationships
    "She felt relieved when the test results came back normal.",
    "He was furious after waiting two hours in line.",
    "They reconciled after months of not speaking.",
    "The child cried after dropping his ice cream.",
    "She smiled nervously before the interview started.",
    "He felt proud watching his daughter graduate.",
    "The breakup left her feeling exhausted and confused.",
    "They laughed together over an old inside joke.",
    "He apologized sincerely for forgetting the anniversary.",
    "She felt a wave of nostalgia walking through her old school.",
    # science
    "The experiment failed to replicate the original results.",
    "Researchers discovered a new species of deep-sea fish.",
    "The vaccine trial entered its final testing phase.",
    "Astronomers detected a faint signal from a distant galaxy.",
    "The chemical reaction released a surprising amount of heat.",
    "Scientists sequenced the genome of an ancient sample.",
    "The telescope captured the clearest image yet of the nebula.",
    "A new study linked sleep quality to memory retention.",
    "The satellite lost contact shortly after launch.",
    "Researchers measured a small but significant effect.",
    # business / finance
    "The company's stock dropped sharply after the earnings call.",
    "They negotiated a lower price after months of talks.",
    "The startup ran out of funding before shipping the product.",
    "Inflation pushed grocery prices higher this year.",
    "The merger was announced without any prior warning.",
    "Investors grew nervous after the unexpected resignation.",
    "The quarterly report exceeded analyst expectations.",
    "They laid off a third of the workforce during the downturn.",
    "The bank raised interest rates for the third time this year.",
    "Sales doubled after the product launch went viral.",
]
