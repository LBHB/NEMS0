# Modules

Modules are simply [pure functions](https://en.wikipedia.org/wiki/Pure_function) now. We highly discourage any side effects, including printing to stdout, caching, or any other stateful action that can escape the bounds of the function. 

Signals should accept a [Recording object](recording.md) and return a Recording object. 

*Proposal that is up for debate:* As a further restriction, modules may accept keyword arguments, but NOT *optional* keyword arguments or keyword arguments with defaults. The goal of this draconian step would be to force developers to make everything explicit in the modelspec so that the whole signal flow diagram is visible when looking at the modelspec. Yes, this means you have to write "input='pred'" for almost every module.


## Making your own module

TODO

