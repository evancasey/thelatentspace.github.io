# Blog

My personal blog. Design heavily inspired by Dustin Tran's [blog](dustintran.com)

## Workflow

Here's my workflow for writing and submitting blog posts.

1. Dump thoughts into a markdown file, in `_drafts/`. Or edit the many
   files already inside `_drafts/`. Preview (and generate) the static
   site from a local server.

  ```bash
  jekyll serve --drafts
  ```
2. When complete, rename and move the file to `_posts/`.
3. Re-build the site.

  ```bash
  jekyll build --destination ../blog
  ```
4. Copy generated blog into VPS.

  ```bash
  scp -r blog ec2:/var/www/ancasey.com
  ```
5. (Optionally), log into VPS and update website.
  (This process can be streamlined by either building on the VPS or
  setting up autogeneration on the VPS and once in a while pulling
  the repo.)

  ```bash
  cd /var/www/ancasey.com
  git pull --all
  ```

## Maintenance

To keep the theme up to date, I track the theme's original repo on
the `type-theme` branch. Add to remote the original repo,
```
git remote add theme git@github.com:rohanchandra/type-theme.git
```
Whenever you want to update, simply run
```
git checkout type-theme
git pull theme master
```
You can compare `type-theme` to `master` and possibly merge in any
changes. Keeping the theme up-to-date on a separate branch avoids
treating the repo as a fork: this repo does more than just style

things and is thus not appropriate as a fork.
