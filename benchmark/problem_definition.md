## Problem definition

This project implements and compares mainly 2 strategies: KIRIS and RAG_GPT on the recognising and coding of medical entities in clinical stroke texts, using SNOMED-CT terminology.

In my machine, I get the following results



    Strategy                       F1-Score   Precision  Recall     Pred   Match  Time

    ------------------------------------------------------------------------------------------------------------------------

    01_KIRIs                       0.8000     0.8381     0.7652     105    88     0.035     s
    04_RAG_GPT                     0.5650     0.5833     0.5478     


My benchmark are clinical sentences, which should give me 32 concepts. I am testing the RAG_GPT with a ontology of those 32 concepts + 2470 extra medical concepts to add sound to the model. I should be able to get at least 0.65-0.7 with the RAG_GPT approach.
