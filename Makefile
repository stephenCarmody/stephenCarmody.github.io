
serve:
	bundle exec jekyll serve --livereload
.PHONY: serve

clean-port:
	lsof -ti :4000 | xargs kill -9 || true
.PHONY: clean_port