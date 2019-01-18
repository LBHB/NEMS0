-- MySQL dump 10.13  Distrib 5.7.23, for Linux (x86_64)
--
-- Host: hyrax    Database: cell
-- ------------------------------------------------------
-- Server version	5.7.22-0ubuntu0.16.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `NarfResults`
--

DROP TABLE IF EXISTS `NarfResults`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `NarfResults` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `cellid` varchar(255) DEFAULT NULL,
  `batch` int(11) DEFAULT NULL,
  `modelname` text,
  `r_test` double DEFAULT NULL,
  `r_fit` double DEFAULT NULL,
  `score` double DEFAULT NULL,
  `sparsity` double DEFAULT NULL,
  `modelpath` text,
  `modelfile` text,
  `figurefile` text,
  `githash` varchar(255) DEFAULT NULL,
  `lastmod` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `r_ceiling` double DEFAULT NULL,
  `r_floor` double DEFAULT NULL,
  `n_parms` int(11) DEFAULT NULL,
  `mi_test` double DEFAULT NULL,
  `mi_fit` double DEFAULT NULL,
  `mse_fit` double DEFAULT NULL,
  `mse_test` double DEFAULT NULL,
  `nlogl_test` double DEFAULT NULL,
  `nlogl_fit` double DEFAULT NULL,
  `cohere_test` double DEFAULT NULL,
  `cohere_fit` double DEFAULT NULL,
  `r_active` double DEFAULT NULL,
  `r_test_rb` double DEFAULT NULL,
  `username` varchar(50) DEFAULT NULL,
  `labgroup` varchar(50) DEFAULT NULL,
  `public` int(11) DEFAULT '0',
  `se_test` double DEFAULT NULL,
  `se_fit` double DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `modelnameidx` (`modelname`(200)),
  KEY `batchidx` (`batch`),
  KEY `cellididx` (`cellid`),
  KEY `batchcellmodelidx` (`batch`,`cellid`,`modelname`(150)),
  KEY `useridx` (`username`),
  KEY `groupidx` (`labgroup`)
) ENGINE=InnoDB AUTO_INCREMENT=1879750 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `NarfBatches`
--

DROP TABLE IF EXISTS `NarfBatches`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `NarfBatches` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `cellid` varchar(255) DEFAULT NULL,
  `est_set` text,
  `val_set` text,
  `filecodes` text,
  `lastmod` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `batch` varchar(255) DEFAULT NULL,
  `est_snr` double DEFAULT NULL,
  `val_snr` double DEFAULT NULL,
  `val_reps` int(11) DEFAULT NULL,
  `est_reps` int(11) DEFAULT NULL,
  `min_snr_index` double DEFAULT NULL,
  `min_isolation` double DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `cellidx` (`cellid`),
  KEY `batchidx` (`batch`)
) ENGINE=InnoDB AUTO_INCREMENT=73503 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tQueue`
--

DROP TABLE IF EXISTS `tQueue`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tQueue` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `rundataid` int(11) DEFAULT NULL,
  `progname` text,
  `parmstring` text,
  `machinename` varchar(255) DEFAULT NULL,
  `pid` int(11) DEFAULT '0',
  `progress` int(11) DEFAULT '0',
  `complete` int(11) DEFAULT '0',
  `queuedate` datetime DEFAULT NULL,
  `startdate` datetime DEFAULT NULL,
  `lastdate` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `killnow` int(11) DEFAULT '0',
  `mailto` varchar(255) DEFAULT NULL,
  `mailcommand` varchar(255) DEFAULT NULL,
  `user` varchar(255) DEFAULT 'david',
  `allowqueuemaster` int(11) DEFAULT '0',
  `computerid` int(11) DEFAULT NULL,
  `priority` int(11) DEFAULT '0',
  `note` varchar(255) DEFAULT NULL,
  `waitid` int(11) DEFAULT '0',
  `extended_status` double DEFAULT '0',
  `memMB` double DEFAULT NULL,
  `GPU_job` int(11) DEFAULT '0',
  `linux_user` varchar(255) DEFAULT NULL,
  `codehash` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `noteidx` (`note`),
  KEY `complete_idx` (`complete`)
) ENGINE=MyISAM AUTO_INCREMENT=383646 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tComputer`
--

DROP TABLE IF EXISTS `tComputer`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tComputer` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL DEFAULT '',
  `load1` double(16,4) DEFAULT '0.0000',
  `load5` double(16,4) DEFAULT '0.0000',
  `load15` double(16,4) DEFAULT '0.0000',
  `lastdate` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `location` int(11) DEFAULT '0',
  `maxproc` int(11) DEFAULT '2',
  `numproc` int(11) DEFAULT '0',
  `maxload` double DEFAULT NULL,
  `allowqueuemaster` int(11) DEFAULT NULL,
  `ext` varchar(255) DEFAULT NULL,
  `owner` varchar(255) DEFAULT NULL,
  `allowothers` int(11) DEFAULT '1',
  `killqueueload` double DEFAULT '1.3',
  `allowqueueload` double DEFAULT '0.3',
  `lastoverload` int(11) DEFAULT '0',
  `pingcount` int(11) DEFAULT '0',
  `dead` int(11) DEFAULT '0',
  `macaddr` varchar(255) DEFAULT NULL,
  `os` varchar(255) DEFAULT NULL,
  `note` varchar(255) DEFAULT NULL,
  `hardware` varchar(255) DEFAULT NULL,
  `room` varchar(255) DEFAULT NULL,
  `nocheck` int(11) DEFAULT '0',
  `maxGPU_jobs` int(11) DEFAULT '0',
  PRIMARY KEY (`id`),
  KEY `nameidx` (`name`)
) ENGINE=MyISAM AUTO_INCREMENT=26 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `NarfAnalysis`
--

DROP TABLE IF EXISTS `NarfAnalysis`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `NarfAnalysis` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `status` varchar(255) NOT NULL,
  `question` text,
  `answer` text,
  `tags` text,
  `batch` varchar(255) NOT NULL,
  `summaryfig` text,
  `lastmod` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `modeltree` text,
  `username` varchar(50) DEFAULT NULL,
  `labgroup` varchar(50) DEFAULT NULL,
  `public` int(11) DEFAULT '0',
  PRIMARY KEY (`id`),
  KEY `useridx` (`username`),
  KEY `groupidx` (`labgroup`)
) ENGINE=InnoDB AUTO_INCREMENT=167 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `NarfData`
--

DROP TABLE IF EXISTS `NarfData`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `NarfData` (
  `dataid` int(11) NOT NULL AUTO_INCREMENT,
  `cellid` varchar(255) DEFAULT NULL,
  `batch` int(11) DEFAULT NULL,
  `groupid` int(11) DEFAULT NULL,
  `label` varchar(50) DEFAULT NULL,
  `filepath` text,
  `notes` text,
  `username` varchar(50) DEFAULT NULL,
  `labgroup` varchar(50) DEFAULT NULL,
  `public` int(11) DEFAULT '0',
  `rawid` int(11) DEFAULT NULL,
  PRIMARY KEY (`dataid`),
  KEY `useridx` (`username`),
  KEY `groupidx` (`labgroup`),
  KEY `cellbatchid` (`cellid`,`batch`)
) ENGINE=InnoDB AUTO_INCREMENT=72127 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `NarfUsers`
--

DROP TABLE IF EXISTS `NarfUsers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `NarfUsers` (
  `id` int(10) NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `password` varchar(255) DEFAULT NULL,
  `email` varchar(50) NOT NULL,
  `firstname` varchar(255) DEFAULT NULL,
  `lastname` varchar(255) DEFAULT NULL,
  `sec_lvl` int(10) NOT NULL DEFAULT '1',
  `labgroup` varchar(50) DEFAULT NULL,
  `selections` text,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2018-10-04 14:37:00
