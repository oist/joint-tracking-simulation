"""
# cross-correlation approach
https://onlinecourses.science.psu.edu/stat510/node/74
https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.correlate.html
https://anomaly.io/detect-correlation-time-series/
http://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.ccf.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html
https://docs.scipy.org/doc/numpy/reference/generated/numpy.correlate.html
https://onlinecourses.science.psu.edu/stat510/?q=book/export/html/75
http://www.statsmodels.org/dev/importpaths.html

neglecting the consideration of autocorrelation, and relying instead on a simple cross-correlation analysis,
may be inadequate and even dangerous in these circumstances
The essential point here is that commonly, meaningless correlations exist between independent pairs of time series
that are themselves autocorrelated.

Independent (non-time-series) data sets that are to be assessed by conventional parametric statistics need to comprise
values that are i.i.d.—that is, independent and identically distributed. In contrast, as we mentioned, time series are
rarely i.i.d., but rather, they are almost always autocorrelated.

Time series that are to be modeled or related to each other are normally molded first to conform within probabilistic
limits to “weak stationarity.” The weak stationarity (routinely simply termed “stationarity”) of a time series is achieved
when the mean and variance are constant and the autocorrelations between values a given number of time points (lags or leads)
apart are also constant. In other words, stationarity does not require the removal of autocorrelation. Thus, a series
with a trend is commonly made stationary by differencing: that is, by constructing a new series (which is one event shorter),
comprising successive differences between adjacent values of the original series (see Sims, 1988, for discussion).
Sometimes other transformations, such as taking the log or the removal of estimated temporal trends using regression-style
procedures, are useful.

the autocorrelation function (acf) comprises the correlation between the values in the series and values at a series
of time lags in the same series

The key function of prewhitening is to remove autocorrelation from at least one of the pair of series under study.
Prewhitening involves decorrelating the putative predictor (referent or control) series, and then applying the same filter
(autoregressive model) required to achieve this to the response series.

Given that the cross-correlations are not necessarily symmetric, the question of the directionality of any possible
influence is not necessarily resolved by the CCF per se.

we suggest that the information in all of the multiple significant CCF lags should be considered

in the remainder of the article we assume that the analytical purpose at hand is mechanistic and goes beyond solely
determining a reliable CCF, toward interpreting its integrated impact.

Granger causality assesses whether the relationship between the two series (now again in their original stationarized
forms, not in their prewhitened forms) is likely to contain a causal element, considered from a statistical perspective.
To be more precise, a variable x can be said to be Granger-causal of variable y if preceding values of x help in the
prediction of the current value of y, given all other relevant information.

The transfer function is the part of the model of the response that expresses the relationship between the input
predictor series and the output series, alongside the autocorrelations, which remain a separate part of the model.


Prewhitening thus may reveal any meaningful cross-correlations. Its fundamental purpose is not just their evaluation,
but rather data exploration: to display the likely leading or lagging relationships, if any, between the pair of series.
The guiding information from both the raw cross-correlations, considered against a realistic significance limit, and
from the prewhitened cross-correlations considered against the ccsl, are used in the analysis of the transfer function
 model relating the two series, as we describe later.


The objective is to understand whether an autocorrelated time series x is predictive of an autocorrelated time series y,
and to produce a model of y comprising its own autoregressive function, the transfer function representing the impact
of x on y, and a white-noise error term (i.e., one that no longer contains any autoregression or any other structured
information). We first stationarize the series (if necessary) and apply the same transformation, usually a single
differencing, to both series. Cross-correlation after prewhitening is then our opening form of data exploration, and
it gives cues as to which lags of x may be influential on y; it may also forewarn us whether there are signs of
bidirectionality that need to be investigated separately.



# cross-recurrence approach
Coco and Dale paper
http://pythonhosted.org/PyRQA/
"""