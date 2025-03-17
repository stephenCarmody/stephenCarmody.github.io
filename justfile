# List available commands
default:
    @just --list

# Serve the Jekyll site with live reload
serve:
    bundle exec jekyll serve --livereload

# Kill any process running on port 4000
clean-port:
    lsof -ti :4000 | xargs kill -9 || true

# Build the site
build:
    bundle exec jekyll build

# Clean the site build
clean:
    bundle exec jekyll clean

# Install dependencies
install:
    bundle install

# Serve with drafts enabled
draft:
    bundle exec jekyll serve --livereload --drafts 