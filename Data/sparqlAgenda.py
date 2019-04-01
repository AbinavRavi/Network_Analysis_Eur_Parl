from SPARQLWrapper import SPARQLWrapper, JSON, TSV, CSV

sparql = SPARQLWrapper("http://linkedpolitics.ops.few.vu.nl/sparql/")
sparql.setQuery("""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX re: <http://www.w3.org/2000/10/swap/reason#>
PREFIX rdfa: <http://www.w3.org/ns/rdfa#>
PREFIX dcterms: <http://purl.org/dc/terms/>

SELECT ?date ?speechnr ?agenda
WHERE { 
  ?sessionday rdf:type lpv_eu:SessionDay .
  ?sessionday dcterms:date ?date.	
  ?sessionday dcterms:hasPart ?agendaitem.
  ?agendaitem dcterms:hasPart ?speech.
  
  ?speech lpv:docno ?speechnr.
  ?agendaitem dc:title ?agenda.
  
  FILTER (?date < "2012-01-01"^^xsd:date) 
  FILTER(langMatches(lang(?agenda), "en"))
  
  } ORDER BY ?speechnr

""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
f=open('queryAgenda.csv','w')
f.write('date|speechnr|agenda\n')

for result in results["results"]["bindings"]:
    f.write('%s|%s|%s\n' % (result["date"]["value"], result["speechnr"]["value"], result["agenda"]["value"]))

f.close()
