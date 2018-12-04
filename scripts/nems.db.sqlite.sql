PRAGMA synchronous = OFF;
PRAGMA journal_mode = MEMORY;
BEGIN TRANSACTION;
CREATE TABLE `NarfResults` (
  `id` integer  NOT NULL PRIMARY KEY AUTOINCREMENT
,  `cellid` varchar(255) DEFAULT NULL
,  `batch` integer DEFAULT NULL
,  `modelname` text
,  `r_test` double DEFAULT NULL
,  `r_fit` double DEFAULT NULL
,  `score` double DEFAULT NULL
,  `sparsity` double DEFAULT NULL
,  `modelpath` text
,  `modelfile` text
,  `figurefile` text
,  `githash` varchar(255) DEFAULT NULL
,  `lastmod` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP 
,  `r_ceiling` double DEFAULT NULL
,  `r_floor` double DEFAULT NULL
,  `n_parms` integer DEFAULT NULL
,  `mi_test` double DEFAULT NULL
,  `mi_fit` double DEFAULT NULL
,  `mse_fit` double DEFAULT NULL
,  `mse_test` double DEFAULT NULL
,  `nlogl_test` double DEFAULT NULL
,  `nlogl_fit` double DEFAULT NULL
,  `cohere_test` double DEFAULT NULL
,  `cohere_fit` double DEFAULT NULL
,  `r_active` double DEFAULT NULL
,  `r_test_rb` double DEFAULT NULL
,  `username` varchar(50) DEFAULT NULL
,  `labgroup` varchar(50) DEFAULT NULL
,  `public` integer DEFAULT '0'
,  `se_test` double DEFAULT NULL
,  `se_fit` double DEFAULT NULL
);
CREATE TABLE `NarfBatches` (
  `id` integer  NOT NULL PRIMARY KEY AUTOINCREMENT
,  `cellid` varchar(255) DEFAULT NULL
,  `est_set` text
,  `val_set` text
,  `filecodes` text
,  `lastmod` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP 
,  `batch` varchar(255) DEFAULT NULL
,  `est_snr` double DEFAULT NULL
,  `val_snr` double DEFAULT NULL
,  `val_reps` integer DEFAULT NULL
,  `est_reps` integer DEFAULT NULL
,  `min_snr_index` double DEFAULT NULL
,  `min_isolation` double DEFAULT NULL
);
CREATE TABLE `tQueue` (
  `id` integer  NOT NULL PRIMARY KEY AUTOINCREMENT
,  `rundataid` integer DEFAULT NULL
,  `progname` text
,  `parmstring` text
,  `machinename` varchar(255) DEFAULT NULL
,  `pid` integer DEFAULT '0'
,  `progress` integer DEFAULT '0'
,  `complete` integer DEFAULT '0'
,  `queuedate` datetime DEFAULT NULL
,  `startdate` datetime DEFAULT NULL
,  `lastdate` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP 
,  `killnow` integer DEFAULT '0'
,  `mailto` varchar(255) DEFAULT NULL
,  `mailcommand` varchar(255) DEFAULT NULL
,  `user` varchar(255) DEFAULT 'david'
,  `allowqueuemaster` integer DEFAULT '0'
,  `computerid` integer DEFAULT NULL
,  `priority` integer DEFAULT '0'
,  `note` varchar(255) DEFAULT NULL
,  `waitid` integer DEFAULT '0'
,  `extended_status` double DEFAULT '0'
,  `memMB` double DEFAULT NULL
,  `GPU_job` integer DEFAULT '0'
,  `linux_user` varchar(255) DEFAULT NULL
,  `codehash` varchar(255) DEFAULT NULL
);
CREATE TABLE `tComputer` (
  `id` integer  NOT NULL PRIMARY KEY AUTOINCREMENT
,  `name` varchar(255) NOT NULL DEFAULT ''
,  `load1` double(16,4) DEFAULT '0.0000'
,  `load5` double(16,4) DEFAULT '0.0000'
,  `load15` double(16,4) DEFAULT '0.0000'
,  `lastdate` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP 
,  `location` integer DEFAULT '0'
,  `maxproc` integer DEFAULT '2'
,  `numproc` integer DEFAULT '0'
,  `maxload` double DEFAULT NULL
,  `allowqueuemaster` integer DEFAULT NULL
,  `ext` varchar(255) DEFAULT NULL
,  `owner` varchar(255) DEFAULT NULL
,  `allowothers` integer DEFAULT '1'
,  `killqueueload` double DEFAULT '1.3'
,  `allowqueueload` double DEFAULT '0.3'
,  `lastoverload` integer DEFAULT '0'
,  `pingcount` integer DEFAULT '0'
,  `dead` integer DEFAULT '0'
,  `macaddr` varchar(255) DEFAULT NULL
,  `os` varchar(255) DEFAULT NULL
,  `note` varchar(255) DEFAULT NULL
,  `hardware` varchar(255) DEFAULT NULL
,  `room` varchar(255) DEFAULT NULL
,  `nocheck` integer DEFAULT '0'
,  `maxGPU_jobs` integer DEFAULT '0'
);
CREATE TABLE `NarfAnalysis` (
  `id` integer  NOT NULL PRIMARY KEY AUTOINCREMENT
,  `name` varchar(255) NOT NULL
,  `status` varchar(255) NOT NULL
,  `question` text
,  `answer` text
,  `tags` text
,  `batch` varchar(255) NOT NULL
,  `summaryfig` text
,  `lastmod` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP 
,  `modeltree` text
,  `username` varchar(50) DEFAULT NULL
,  `labgroup` varchar(50) DEFAULT NULL
,  `public` integer DEFAULT '0'
);
CREATE TABLE `NarfData` (
  `dataid` integer NOT NULL PRIMARY KEY AUTOINCREMENT
,  `cellid` varchar(255) DEFAULT NULL
,  `batch` integer DEFAULT NULL
,  `groupid` integer DEFAULT NULL
,  `label` varchar(50) DEFAULT NULL
,  `filepath` text
,  `notes` text
,  `username` varchar(50) DEFAULT NULL
,  `labgroup` varchar(50) DEFAULT NULL
,  `public` integer DEFAULT '0'
,  `rawid` integer DEFAULT NULL
);
CREATE TABLE `NarfUsers` (
  `id` integer NOT NULL PRIMARY KEY AUTOINCREMENT
,  `username` varchar(50) NOT NULL
,  `password` varchar(255) DEFAULT NULL
,  `email` varchar(50) NOT NULL
,  `firstname` varchar(255) DEFAULT NULL
,  `lastname` varchar(255) DEFAULT NULL
,  `sec_lvl` integer NOT NULL DEFAULT '1'
,  `labgroup` varchar(50) DEFAULT NULL
,  `selections` text
);
CREATE INDEX "idx_NarfData_useridx" ON "NarfData" (`username`);
CREATE INDEX "idx_NarfData_groupidx" ON "NarfData" (`labgroup`);
CREATE INDEX "idx_NarfData_cellbatchid" ON "NarfData" (`cellid`,`batch`);
CREATE INDEX "idx_tQueue_noteidx" ON "tQueue" (`note`);
CREATE INDEX "idx_tQueue_complete_idx" ON "tQueue" (`complete`);
CREATE INDEX "idx_tComputer_nameidx" ON "tComputer" (`name`);
CREATE INDEX "idx_NarfResults_modelnameidx" ON "NarfResults" (`modelname`);
CREATE INDEX "idx_NarfResults_batchidx" ON "NarfResults" (`batch`);
CREATE INDEX "idx_NarfResults_cellididx" ON "NarfResults" (`cellid`);
CREATE INDEX "idx_NarfResults_batchcellmodelidx" ON "NarfResults" (`batch`,`cellid`,`modelname`);
CREATE INDEX "idx_NarfResults_useridx" ON "NarfResults" (`username`);
CREATE INDEX "idx_NarfResults_groupidx" ON "NarfResults" (`labgroup`);
CREATE INDEX "idx_NarfBatches_cellidx" ON "NarfBatches" (`cellid`);
CREATE INDEX "idx_NarfBatches_batchidx" ON "NarfBatches" (`batch`);
CREATE INDEX "idx_NarfAnalysis_useridx" ON "NarfAnalysis" (`username`);
CREATE INDEX "idx_NarfAnalysis_groupidx" ON "NarfAnalysis" (`labgroup`);
END TRANSACTION;
