# wilfried library

This library contains a set of sub-libraries with different goals which were/are developed during my internship/phd at IRAP.

The main functions are either related to galaxies or to making it easier to plot complex graphs.

Here is a summary of the diffrent sub-libaries:

## **galaxy.py**

A set of functions related to computing galaxy properties such as half-light radii, luminosity profiles, intensities, etc.

## **galfit.py**

A set of functions used to automatise the use galfit for a large number of galaxies.

## **plots**

Plotting related functions.

### **plotUtilities**

Functions useful to simplify plotting complex graphs with matplotlib. The main function asManyPlots is a general and highly tunable function used to easily make nice, complex plots with numerous data.

## **makeLifeSimple.py**

Miscellanous functions used in data processing to speed things up.

## **strings**

Strings related functions. This was detached from makeLifeSimpler.py since it does not need any numpy or scipy function to be used.

So any user who might be interested to use these functions only, can directly import any library from this module.

However, these functions are also imported in makeLifeSimpler. So any import of makeLifeSimpler will automatically import the strings functions as well.

### ***strings***

The library with all the strings functions.
