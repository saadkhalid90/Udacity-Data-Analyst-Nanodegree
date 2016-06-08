## Summary:
In this project, I explored the titanic dataset and constructed an explanatory data visualization that conveys the following message about data.

“The percentage for survival for first class passengers was greater than the second, which in turn was greater than that of the third. Moreover, overall survival rate for men was lesser than than that for women in all passenger classes.”  

## Design:
The link to my initial sketch is:
http://bl.ocks.org/saadkhalid90/raw/2c61d597cb78676eddff/

In this graphic, I was using y-position of the points as visual encoding for male and female survival rate. My thinking behind connecting the male and female points with a line in each class was aimed to emphasize the difference between male and female survival rates (slope and tilt as an indicator of difference). But based on the feedback, I had to get rid of these visual encodings which seemed to be causing confusion for many readers. 

As a follow up, I changed the visual encodings and tried height of simple bars to represent male and female survival rates for each class. The different classes were represented across the x-axis and male and female bars were colored differently. The link to the changes visualization is:

http://bl.ocks.org/saadkhalid90/raw/68ea2c147282f1219d6d/

Some changes were also made upon feedback following the second iteration. I used text to also indicate the percentage rates on the bars. Moreover, I also customized the default hover feature offered by ‘dimple.js’, as it was showing approximate rather than actual values for survival rates. For implementation these design decision, I used dimple.js to construct the chart but also used functionality of d3.js to customize my chart based on reader feedback. The link to the final visualization is:

http://bl.ocks.org/saadkhalid90/raw/79c5798ebdf070b9e986/

I kept the default colors implemented by dimple.js as I personally found them simple and aesthetically pleasing 

## Feedback:
I used the Udacity nanodegree Google+ group as a forum to get feedback from fellow students. Moreover, I also emailed fellow grad students at my university for feedback. For my first sketch, Major points that summarize the feedback are as follows.

- The connecting line between Female and male survival rates is misleading because it gives an idea that the two rates are related in some way which is not the case. This can be a source of confusion for the reader.
- The plot appears empty and can utilize other visual encodings to express the same information more effectively.
- Although the visualization conveys the underlying idea, it requires too much eye movement and effort on the part of the reader to fully understand it.    

I incorporated the feedback and used bars to encode survival rates for females and males across different passenger classes. At this point, I shared my visualization again to get the following feedback:
- The default hover functionality in dimple.js was showing rounded off rates (e.g a rate of 0.97 was being shown as 1.0 and a rate of 0.16 was rounded off to 0.2).
- Additionally, a reader also suggested adding text on the bars depicting the rates in percent for helping the reader understand the visualization.

In light of this feedback, I customized the default hover functionality by adding information represented by each bar at the top right of the visualization if the user hovers on a bar. Moreover, I also used d3 to append text elements to each bar that indicate the percentage rates associated with each bar. 

## Resources:
- https://www.udemy.com/data-visualize-data-with-d3js-the-easy-way/
- https://github.com/PMSI-AlignAlytics/dimple/wiki/dimple
- http://dimplejs.org/





    


