import tornado.ioloop
import tornado.web
import os
from sa import SentimentAnalysis as SA

model = SA.SentimentAnalysis('sa/movie-reviews-sentiment.tsv')
model.fit();
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        query = self.get_argument("query", "").strip()
        result = model.predict([query])
        if len(query) > 0:
            print result[0]
            self.write(result[0])
        else:
            self.render("index.html")
            self.set_header("Cache-Control","no-cache")


if __name__ == "__main__":
    print "Done building model"
    dirname = os.path.dirname(__file__)
    settings = {
        "static_path" : os.path.join(dirname, "static"),
        "template_path" : os.path.join(dirname, "template")
    }
    application = tornado.web.Application([
    (r"/", MainHandler)], **settings)

    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
