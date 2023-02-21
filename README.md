#The Search for Planet X solver simulator

This repo contains code that:

* Generates possible boards for the board game "The Search for Planet X"
* Scores possible survey actions in a few different ways
* Simulates completing a search using each strategy many times

It's not really in a usable state, I was just playing around. Instructions to run this may or may not come at some point.

Possible future additions:

* A user-input driven solver, allowing you to input search results from a real game
* Research! What's the expected information from a picking a research project?

# About research

I have a script that extracts research clues from some PDFs I found on BGG, but it's not straightforward to add these into the solver, because we don't have research clues for all >4000 possible boards. This is important because the game creators explicitly try to avoid clues that give away too much e.g. Planet X is next to a comet, when the comets are adjacent to each other.

At best, this'll let us estimate how good research is, on average. That might let you decide when to pick research in a real game, but because I don't know how to generate clues for boards, we can't simulate the research action.
