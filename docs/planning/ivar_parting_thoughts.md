# Parting Thoughts on NEMS Development
 
Since remaining development on NEMS is going to have to be done largely without me, I thought I would try to write out the areas of NEMS where the architecture is not completely done yet. 

## REPOSITORY ORGANIZATION

We made a lot of progress with repository organization! Instead of one giant repo to rule them all, we now have four smaller repos, each with a specific purpose. This is a mild annoyance at times if you are used to doing everything in a single repo, but it really does help reduce the amount of code you have to search through when there is a bug. To review, here are the four repositories and their purposes as I conceive of it. I don't mean this to be the final word on the subject -- feel free to move things around as you see fit, perhaps carefully combining the nems_db and nems_web repos together -- but I do think it is important for me to try to clarify what my goals and thoughts were for each repository:

    1. NEMS. We made a lot of progress making this a minimalist repo that specializes in fitting a model to some data. It takes in one or more Recordings (as files), defines a model, fits the model, and spits results out to a directory somewhere. We should try to keep it agnostic as to how data is stored or organized, so that other labs can simply download this repo and use it.
       Repo TODO Items:
       - [ ] Make the 'master' branch into the 'old-master' branch
       - [ ] Move the 'dev' branch to the 'master' branch, removing the large files from the 'signals/' directory and making the master branch history not contain large files anymore. NEMS master should be only a 20k download of code, not hundreds of megabytes of sample data.
       - [ ] Restrict pushes to the "master" branch to trusted developers only
       - [ ] Make a LBHB branch of nems, which anyone in the lab can push to (instead of the master branch)
       - [ ] Write a tutorial on how to merge parts of the LBHB branch back into master
       - [ ] Write a tutorial on how to fork the master branch, and how to do a pull request, so that other labs can safely contribute back to NEMS

    2. NEMS_DB. I conceive of this as mostly being a file repository for storing manually-generated recordings, and for storing results. Ideally, this repository stays simple enough that other labs can set up their own NEMS_DB's to share results within their lab, or also with LBHB. It also seems like the logical place to connect to a SQL table in which all of the fitting results and performance metrics are stored, along with a simple API that returns the URIs of the things you want to load.
       - [ ] Write instructions on how external labs can access our NEMS_DB server by using SSH to forward http://hyrax.ohsu.edu:3000 to their own localhost:3000, and then save files to our NEMS repo. This would let other labs securely save files with us.
       - [ ] NEMS clients can currently directly download recordings from the public S3 bucket; this should be where we put our 'sample' data for other labs to use. However, NEMS clients cannot upload to S3 right now. I think it makes sense to have the NEMS_DB server manage uploads to S3, so that the S3 credentials need only live in one place. This should not be too hard to write, and basically involves writing another "PUT" HTTP endpoint on the NEMS_DB server. 
       - [ ] Write instructions for beginners on launching their own nems_db server
       - [ ] Write instructions on how to use rsync to merge the 'results/' directories from two different nems_db repositories.

    3. NEMS_BAPHY. This is the place to put anything specific to getting data out of Baphy. I think it should also be the place where "batches" are maintained; an API route that lets you get a list of every cellid in a batch would probably be a good idea. As for work tickets:
       - [ ] I'm concerned that baphy.py is going to silently fail when given bad optional arguments, and that this will be a pain for people to debug. Wrapping the calls to baphy.py in try/catch, and then returning a helpful error message to the clients with a 400 or 500-series HTTP error code. During debugging, NEMS_BAPHY shoud probably be run locally -- I would not recommend trying to develop nems_baphy or fix bugs "in place" on hyrax.
       - [ ] There need to be better 'options" defaults in place for baphy.py, so that going to http://hyrax.ohsu.edu/baphy/271/TAR010c-11-1 will work even without any optional arguments provided. More importantly, Stephen needs to write documentation for what arguments are required for what batch, as there seems to be a lot of batch-specific arguments. 
       - [ ] I am mildly concerned that the NGNIX timeouts (which are 15 seconds by default? I tried to increase this) may need to be adjusted if generating Recording objects takes a very long time in some cases.
       - [ ] Profiling nems_baphy and optimizing it would probably dramatically decrease the Recording generation runtimes.
       - [ ] The NGINX caching of calls to nems_baphy works; however, if nems_baphy has an error, then NGINX also caches those errors! I believe you can adjust NGINX to not cache errors, but I'm not sure I set it up correctly. 
       - [ ] The NGINX cache is currently in the /home/nems/nginx_cache/ directory on hyrax (not on /auto/data/tmp). You can erase the contents of this directory at any time to clear the cache. 

    4. NEMS_WEB. A lot of the interface here does not need to change, but my hope is that serving out static files from NEMS_DB results/ will make showing multiple plots simpler. It also may make sense for nems_web to `import nems_db` and `import nems_baphy` so that NEMS_WEB can directly query any relevant MySQL tables, without duplicating any source code. Let's try to put the days of executing modelspecs on the server behind us; it's a little too error-prone a process to ever do safely.

    5. NEMS_QUEUE. Given more time, I would have tried to stub out a clustering API that uses Celery and a Redis or RabbitMQ backend. The main advantage here is that these systems scale up larger than our lab's homegrown clustering system, and are more suited to deployment in the cloud. It's not hard to imagine writing a simple script for AWS EC2 instances that 1) downloads NEMS from github, 2) sets up the Celery client and connects to the LBHB lab queue; and 3) starts running NEMS jobs that were queued up here in the lab. This would let you do hundreds of NEMS fits simultaneously. 

## OTHER ARCHITECTURAL ISSUES

NONLINEARITIES AND PREFITS. I was hoping to get away from the "prefitting" concept entirely, and instead provide just "priors", but we clearly are not there yet. I don't have any philosophical objections to "prefitting", because I know that it works, but I think everybody would be happier, we could test a wider varitey of off-the-shelf fitting algorithms, and our fits would finish faster and more reliably if we didn't have to do prefits.

So, how do we get rid of prefits? I don't exactly know, but I think we would benefit from a careful study of what sort of parameters are found by our fitting algorithms across the entire population, so that we can define "priors" that actually start the fit in the areas most likely to be good.

But, at the end of the day, if we really can't get rid of prefits, then at least xforms provides a way for us to record how they were done.

NEMS_DB TREE ORGANIZATION. I would /not/ making things here very deeply nested. Right now, the separation is /recordingname/modelkeywords/xformkeywords/date/, and I think this is really about as simple as we can make it. The advantage of this simple choice is that all four fields will _always_ be defined. Adding a nesting layer like "batch" is problematic because there may be data sets that are not part of any batch. Instead, I would advocate making batch number part of the recordingname. Perhaps this should be done when the recording is generated by nems_baphy?

MEMOIZATION AND OPTIMIZITION. We could probably get a 2x speedup in model fits by using joblib's memoization system in nems.modelspec.evaluate(). To make sure we do it right, the optimization must be done with careful before-and-after benchmarks.

CONDA. I'm not a conda expert, but based on what Brad showed me, I updated the notes in docs/conda.md. I highly recommend that you use a conda environment with the Intel MKL libs installed, as it is about 2x faster on my machine than the default linux python distribution.

PER-MODULE PLOT FUNCTIONS. I would recommend never adding per-module plot functions or other module-specific fields. The reason why is that many times you may want to plot zero, one, or many different plots of the parameters of a module, and sometimes aggregate across jackknifes, and each modelspec cannot always know the context in which it is being asked to generate a plot. Instead, treat plotting as ways of creating a "view" of some number of modelspecs; each plot function can create a different view as needed, and you can always make new views at any time. In NARF and in the old NEMS, modelspecs were a little too "executable" and caused a lot of magic things to occur when plotting. In the new version of NEMS, let's strive to keep them just as data without any executable code attached to them. Embedding plotfns in every single modelspec also unnecessarily increases the size of every modelspec.

XFORMS KEYWORDS. Rather than add a second layer of indirection like we did for the modelspec keywords, we should probably just make the names of the xforms functions short. I can't think of anything simpler right now.

SPARSIFIED SIGNAL OBJECTS. Someday, it would be useful to subclass Signal objects and have them store "spike event times" so that resp could be an unrastered list of spike events. But this is probably a week of work to implement and test, so it may be best to do this only when an experiment or scientific result actually demands that it be written.

PARAMETER VARIANCE ESTIMATION. Brad convinced me that we should try AVDI (http://www.stat.columbia.edu/~gelman/research/unpublished/advi_journal) like in PyMC3 to to estimate the posterior parameter distribution. Essentially, this involves making an assumption about the parameter's posterior distribution (gaussian), and then fitting the mu/sigma statistics by sampling a few times to get the MSE. This requires many fewer samples than Monte Carlo methods. 

ADVANCED FITTERS. I never got around to implementing any, but the current fitting function architecture should support fitters that only fit subsets of the parameters or subsets of the data on each fit step. So whenever that day comes, you shouldn't need to make any new architecture -- just a fitter! 
