<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#sec-1">1. C++ Project Template</a>
<ul>
<li><a href="#sec-1-1">1.1. Introduction</a></li>
<li><a href="#sec-1-2">1.2. How to use</a>
<ul>
<li><a href="#sec-1-2-1">1.2.1. Clone or fork or download</a></li>
<li><a href="#sec-1-2-2">1.2.2. Change the default project name</a></li>
<li><a href="#sec-1-2-3">1.2.3. Use it as you wish</a></li>
</ul>
</li>
<li><a href="#sec-1-3">1.3. Requirements</a></li>
<li><a href="#sec-1-4">1.4. How to contribute</a></li>
<li><a href="#sec-1-5">1.5. Licence</a></li>
</ul>
</li>
</ul>
</div>
</div>

# C++ Project Template<a id="sec-1" name="sec-1"></a>

A C++ Project Template. 
Because you should concentrate on the important parts.

## Introduction<a id="sec-1-1" name="sec-1-1"></a>

This is a minimal, blank cmake project for C++.
Making files, folders, CMakeLists.txt and all the boring stuff: done and ready to go.

## How to use<a id="sec-1-2" name="sec-1-2"></a>

### Clone or fork or download<a id="sec-1-2-1" name="sec-1-2-1"></a>

-   Clone this project and use it as you wish.
    The project will appear as cloned.

``` shell
    git clone https://github.com/Red-Portal/project-template.git
    cd project-template

```

-   Fork this project and use it. If you are really lazy.
    The project will appear as forked

-   Download and unzip the project.

### Change the default project name<a id="sec-1-2-2" name="sec-1-2-2"></a>

change all occurences of `template-project` to your project name!
Run the install script as below in order to do this automatically.
The install script will self delete upon success.

``` shell
	sudo sh ./install.sh --NAME=PROJECT_NAME
```

Windows powershell script will be added soon

### Use it as you wish<a id="sec-1-2-3" name="sec-1-2-3"></a>

-   Add custom CMake Find modules to `./cmake`
-   Add header interfaces to `./include/PROJECT_NAME/`
-   Add source files to `./src`
-   Add documentations to `./doc`

## Requirements<a id="sec-1-3" name="sec-1-3"></a>

1.  git
2.  C++ compiler
3.  CMake

## How to contribute<a id="sec-1-4" name="sec-1-4"></a>

This project is open for contribution.
Send me push requests any time you have a good idea.
Please describe reasons for additions or fixes.

## Licence<a id="sec-1-5" name="sec-1-5"></a>

MIT License

Copyright (c) 2017 Red-Portal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
