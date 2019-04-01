from SPARQLWrapper import SPARQLWrapper, JSON, TSV, CSV

sparql = SPARQLWrapper("http://linkedpolitics.ops.few.vu.nl/sparql/")
sparql.setQuery("""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX re: <http://www.w3.org/2000/10/swap/reason#>
PREFIX rdfa: <http://www.w3.org/ns/rdfa#>
PREFIX dcterms: <http://purl.org/dc/terms/>

SELECT ?date ?speechnr ?name ?nationality ?text 
WHERE { 
  ?sessionday rdf:type lpv_eu:SessionDay .
  ?sessionday dcterms:date ?date.	
  ?sessionday dcterms:hasPart ?agendaitem.
  ?agendaitem dcterms:hasPart ?speech.
  
  ?speech lpv:docno ?speechnr.
  ?speech lpv:translatedText ?text.
  ?speech lpv:speaker ?speaker.
  ?speaker lpv:name ?name.
  
  ?speaker lpv:countryOfRepresentation ?country.
  ?country rdfs:label ?nationality.
  
  FILTER (?date < "2012-01-01"^^xsd:date) 
  FILTER(langMatches(lang(?text), "en"))
  #FILTER(?nationality="United Kingdom"@en)
  
  } ORDER BY ?speechnr

""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
f=open('queryName.csv','w')
f.write('date|speechnr|name|nationality|text\n')

for result in results["results"]["bindings"]:
    f.write('%s|%s|%s|%s|%s\n' % (result["date"]["value"], result["speechnr"]["value"], result["name"]["value"], result["nationality"]["value"], result["text"]["value"].replace("\n"," ").replace("|","")))

f.close()
