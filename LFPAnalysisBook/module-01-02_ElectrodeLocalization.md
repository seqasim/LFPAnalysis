# Electrode Localization

Critically, this Python codebase  can do many things related to analyzing your iEEG data, but it cannot localize the electrodes in the brain. It assumes you've done this on your own outside of this analysis pipeline. 

There are many options for doing this. This notebook largely requires electrode information (labels, mni coordinates, regions) in the form of a pandas dataframe - a goal that can be achieved by basically any localization solution. Here, we largely assume the use of [LeGUI](https://github.com/Rolston-Lab/LeGUI), which is the best tool for this in my opinion. 

But again, I emphasize - any tool you use that outputs a csv or excel of [labels, coordinates, atlas regions] should play nice with this codebase (with minor edits). 
