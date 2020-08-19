# Unlocking Insights from Unstructured Data with Text Mining

- 10th December 2019
- [BI - General](https://www.peakindicators.com/ev_blog/blog_categories/view/17)
- Paul Clough and Felicity Borg

**Executive summary**

Many, if not all, organisations store and manage unstructured data in the form of text, such as employee emails, invoices, company reports, intranet pages, social media data, survey responses, etc. However, compared to numeric data, unlocking insights from textual data can be difficult as traditional data analysis methods are not readily applicable. Text mining (extracting meaningful information from text) and text analytics makes use of techniques from machine learning and natural language processing to pre-process and analyse text. In this article, we briefly summarise text mining and discuss an example client use case to analyse vehicle damage costs. 

**Unstructured text data** 

According to predictions by IDC described in their report [“The Digitization of the World: From Edge to Core”](https://www.seagate.com/files/www-content/our-story/trends/files/idc-seagate-dataage-whitepaper.pdf) the total volume of data will grow from 33 zettabytes in 2018 to 175 zettabytes in 2025. It is expected that 80% of this will be [‘unstructured’ ](https://www.datamation.com/big-data/structured-vs-unstructured-data.html)data and much of this will be text - this represents the largest data source produced by people and therefore a rich source of data for analytics and deploying AI applications in the enterprise. However, companies are often only used to managing and analysing ‘structured’ data that fits neatly within the rows and columns of a database. To handle unstructured text data requires the use of approaches, such as text mining.

 **A brief overview of text mining** 

With advances in technology computers are able to read, understand, use and even interpret human language, which supports activities such as text mining and analytics. Similar to data mining, the purpose of [text mining](https://en.wikipedia.org/wiki/Text_mining) is “the discovery by computer of new, previously unknown information, by automatically extracting information from different written resources” or more succinctly expressed in [this article ](https://arxiv.org/abs/1707.02919)as “the task of extracting meaningful information from text.”![img](https://www.peakindicators.com/images/get_image/935/) 

**Figure 1**: Knowledge Discovery in Databases (KDD) process.

Conceived in 1995 from within the Knowledge Discovery in Databases (KDD) community, text mining draws upon several related fields including information retrieval, computational linguistics, data mining, and perhaps the most closely related field - *natural language processing* (NLP), a sub-field of AI which aims to process and understand natural language using computers. The process of knowledge discovery is often an iterative and exploratory one, often following new questions based on insights gained from the findings of prior stages of analysis. 

Text mining supports the broader knowledge discovery process (see Figure 1) that goes from raw data at one end to developing human knowledge at the other through the stages of data selection and processing, transformation and identifying patterns – these can be automatically derived from data using text mining methods, or shown graphically to users through data visualisations and reports.

![img](https://www.peakindicators.com/images/get_image/953/)

**Figure 2**: Typical text mining process (adapted from: [Text Analysis](http://rpubs.com/bnevt0/AT336106))

The process for text mining (shown in Figure 2) is similar and starts with gathering raw text (e.g. using search engines or extracted from databases or APIs), pre-processing and cleaning the text (e.g. segment character strings into words and phrases, removing redundant or common words, identifying parts-of-speech, etc.), then transforming it into a numeric representation that can be input to data mining or analytics processes (e.g. word vectors or a document-term matrix).

Commonly, statistical and machine learning approaches (especially ‘deep learning’ methods) are utilised within the text mining process as these often perform well on large amounts of data and can model characteristics of natural language, such as word co-occurrences and ambiguity.

**Applications of text mining** 

Natural language processing and text mining can be used to develop various applications that may support the activities of individuals and businesses. For example, automatically filtering junk email, identifying hate speech in social media posts, helping people search and navigate large document collections, developing question answering services (e.g. chatbots), automatically classifying and translating text, identifying sentiment or opinion in customer feedback, and extracting key facts or entities (e.g. people, places, organisations, products, etc.) from corporate data.

The process of text mining and analysis can also be a valuable way of organisations gaining insights into key business problems or issues. For example, a 2018 report by SAS entitled [“What can text analytics do for your organisation?” ](https://www.sas.com/content/dam/SAS/en_us/doc/whitepaper1/text-analytics-for-executives-109630.pdf)highlights a number of business use cases where text analytics and mining can be applied including “detecting and tracking service or quality issues, quantifying customer feedback, assessing risk, improving operational processes, enhancing predictive models and many more.” This includes examples in various sectors, ranging from Government, Manufacturing, Financial Services, and Healthcare.

By way of example consider *information extraction* (automatic extraction of structured data or facts). Figure 2 shows an application by [TextRazor](https://www.textrazor.com/) that, amongst other things, identifies entities within text and links these to established knowledge bases - in this case Wikipedia and DBPedia. In this example, the word ‘Belfast’ is recognised as the category type Place and linked to the relevant knowledge sources (e.g. Belfast, Ireland in Wikipedia). In an organisation this could, for example, be used to extract products from written text (e.g. blog posts) and linked to a product catalogue.

 ![img](https://www.peakindicators.com/images/get_image/937/)

**Figure** **2**: Information extraction from example text using TextRazor (source:[ https://www.textrazor.com/demo](https://www.textrazor.com/demo))[)](https://www.textrazor.com/demo)).

Computational text mining is particularly useful for handling large volumes of text where extracting key information manually would be infeasible. For example, a 2018 report from Deloitte entitled “[Using AI to unleash the power of unstructured government data](https://www2.deloitte.com/content/dam/Deloitte/lu/Documents/public-sector/lu-ai-unstructured-government-data.pdf)” highlights the need for automated text processing and analytics capability in the large amounts of textual data generated by the US federal government (to reach 500 million pages in 2024). The use of text analytics and natural language processing would allow government agencies to “recognise patterns, categorise topics and analyse public opinion.” 

 ![img](https://www.peakindicators.com/images/get_image/938/) 

**Figure** **3**: Sentiment analysis example using Dandelion (source: https://dandelion.eu/).

Another example of applying text mining would be for a business to gain insights into what customers think about the products or services they provide. This could be achieved through *sentiment analysis*, whereby text can be analysed to establish the likely opinion of its authors. Figure 3 shows an example using the [Dandelion semantic text analysis service](https://dandelion.eu/) for sentiment analysis of customer reviews of the Disney Plus service. Each review is analysed and a score (between -1 and +1) assigned to indicate the degree of positive (+) or negative (-) sentiment. The first example is particularly interesting as it includes some of the biggest challenges of dealing with natural language – sarcasm, irony and ambiguity – that can cause computational methods to perform poorly. 

**Popular tools** 

Various tools and libraries are available for processing and analysing natural language texts, including commercial 3rd party software that incur license costs (e.g. TextRazor and Dandelion). The major providers of cloud-based services, such as Microsoft Azure and Amazon AWS, also incorporate text mining and analytics functionality in their offerings, for example [cognitive services](https://azure.microsoft.com/en-gb/services/cognitive-services/) in Azure.

In comparison, [open source alternatives](https://blog.dominodatalab.com/comparing-the-functionality-of-open-source-natural-language-processing-libraries/) are becoming widely used, which typically cost nothing to obtain. Open source tools and libraries for text mining and natural language processing (e.g. SparkNLP, SpaCy, NLTK and CoreNLP) have gained popularity in recent years due to the use of state-of-the-art technologies (e.g. the use of ‘deep learning’ methods), production-grade performance and their compatibility with popular languages, such as Python, Scala and Java. 

See this [link](https://www.peakindicators.com/blog/state-of-the-art-natural-language-processing-with-python-and-spacy) for more information about natural language processing and the state-of-the-art Python open source library SpaCy. 

**Example Peak project – Northgate** 

Recently at Peak Indicators we have been investigating the use of text mining and analytics for [Northgate Plc](https://www.northgatevehiclehire.co.uk/) – the UK’s largest commercial vehicle rental provider, with over 100,000 vehicles in the UK, Ireland and Spain. We have utilised [SpaCy](https://spacy.io/), a state-of-the-art Python-based natural language processing framework for processing and extracting information from text.

***"Having engaged Peak Indicators to support us with historic data sets, we were extremely pleased with their ability to manage our expectations from start to finish. Their professional approach and machine learning outputs will play a vital role in helping to formulate Northgate’s 2020 data strategy!"\***

*James Coughlan - Northgate plc*

The aim of the work is to determine the feasibility of applying text mining and analytical methods to provide deeper insights. Northgate collect both structured and unstructured data related to its vehicles, such as work carried out and subsequent invoicing, some of which has been used as the basis for this analysis.

Using a sample dataset, we have been able to identify entities from free text within fields of the dataset and have cross-referenced this against cost(s). This is enabling the business to address previously unanswered questions and gain a flurry of new insights.

**Want to find out more?** 

If you want to find out more about how Peak Indicators can help you get more value and insights from your data, then please [contact us](https://www.peakindicators.com/contact). We have experience with using cloud-based systems, such as Microsoft Azure, and can offer [training](https://www.peakindicators.com/training) in topics such as data science, text mining, [machine learning](https://www.peakindicators.com/service/tallinn), and [data](https://www.peakindicators.com/service/analytics)[ analytics](https://www.peakindicators.com/service/analytics). 