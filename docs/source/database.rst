Database
========

Overview
--------

*Placeholder* table to be turned into description of database

+-------------+--------+--------------------------------------------------------+
| Field       | Type   | Description                                            |
+=============+========+========================================================+
| id          | str    | Name of the module keyword that created this dict.     |
+-------------+--------+--------------------------------------------------------+
| fn          | str    | The pure transformation function.                      |
+-------------+--------+--------------------------------------------------------+
| fn\_kwargs  | dict   | Non-fittable arguments that are passed to fn.          |
+-------------+--------+--------------------------------------------------------+
| meta        | dict   | A dictionary of metadata, descriptions, benchmarks,    |
|             |        | etc                                                    |
+-------------+--------+--------------------------------------------------------+
| phi         | dict   | Like fn, but fittable. The "parameters" of the module. |
+-------------+--------+--------------------------------------------------------+
| prior       | dict   | Defines a prior distribution from which bounds or      |
|             |        | samples may be drawn by the fitter during              |
|             |        | optimization.                                          |
+-------------+--------+--------------------------------------------------------+

Options
-------

sqlite default

mysql large scale

central database?
