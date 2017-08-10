import HTMLParser,sys,os,string


tagstack = []
class ShowStructure(HTMLParser.HTMLParser):
	def handle_starttag(self, tag, attrs): tagstack.append(tag)
	def handle_endtag(self, tag): tagstack.pop()
	def handle_data(self, data):
		if data.strip():
			for tag in tagstack: sys.stdout.write('/'+tag)
			#you could use data here, if you want more character
			sys.stdout.write(' >> %s\n' % data[:40].strip())



def printZhFeed(feedaddr):
	import feedparser
	ny = feedparser.parse(feedaddr)
	if len(ny['entries'])  == 0:
		print 'no feed or connect fail.'
		return

	s = ny['entries'][1].summary_detail['value']	
	ShowStructure().feed(s)
	#To check the source, just print s
	return

