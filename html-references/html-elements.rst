HTML Elements
--------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - HTML Element
     - Description
   * - ``<html>``
     - Root element of an HTML document.
   * - ``<head>``
     - Contains meta-information about the document.
   * - ``<title>``
     - Sets the title of the document.
   * - ``<meta>``
     - Defines metadata about an HTML document (e.g., character set, description).
   * - ``<link>``
     - Defines the relationship between the current document and an external resource (e.g., stylesheets).
   * - ``<style>``
     - Contains style information for the document. 
       Example: ``<style> body { background-color: lightblue; } </style>``
   * - ``<script>``
     - Contains or references executable script. 
       Example: ``<script> console.log('Hello, world!'); </script>``
   * - ``<body>``
     - Contains the visible content of the document. 
       Example: ``<body> <h1>Hello, world!</h1> </body>``
   * - ``<header>``
     - Defines a header for a document or section. 
       Example: ``<header> <h1>Header</h1> </header>``
   * - ``<footer>``
     - Defines a footer for a document or section. 
       Example: ``<footer> <p>Copyright © 2024</p> </footer>``
   * - ``<nav>``
     - Defines navigation links. 
       Example: ``<nav> <a href="#home">Home</a> | <a href="#about">About</a> </nav>``
   * - ``<main>``
     - Specifies the main content of a document. 
       Example: ``<main> <h1>Main Content</h1> <p>Content goes here...</p> </main>``
   * - ``<article>``
     - Defines an article. 
       Example: ``<article> <h2>Article Title</h2> <p>Article content...</p> </article>``
   * - ``<section>``
     - Defines a section in a document. 
       Example: ``<section> <h2>Section Title</h2> <p>Section content...</p> </section>``
   * - ``<aside>``
     - Defines content aside from the content it is placed in. 
       Example: ``<aside> <h4>Aside Content</h4> <p>Additional content...</p> </aside>``
   * - ``<h1>`` to ``<h6>``
     - Defines HTML headings. ``<h1>`` is the largest, ``<h6>`` is the smallest. 
       Example: ``<h1>Heading 1</h1>``
   * - ``<p>``
     - Defines a paragraph. 
       Example: ``<p>This is a paragraph.</p>``
   * - ``<a>``
     - Defines a hyperlink. 
       Example: ``<a href="https://www.example.com">Visit Example</a>``
   * - ``<img>``
     - Embeds an image. 
       Example: ``<img src="image.jpg" alt="Image description">``
   * - ``<ul>``, ``<ol>``, ``<li>``
     - Defines lists. ``<ul>`` for unordered, ``<ol>`` for ordered, and ``<li>`` for list items. 
       Example: ::
       
         <ul>
           <li>Item 1</li>
           <li>Item 2</li>
         </ul>
       
   * - ``<dl>``, ``<dt>``, ``<dd>``
     - Defines a description list, terms, and descriptions. 
       Example: ::
       
         <dl>
           <dt>Term 1</dt>
           <dd>Description 1</dd>
           <dt>Term 2</dt>
           <dd>Description 2</dd>
         </dl>
       
   * - ``<figure>``, ``<figcaption>``
     - Defines self-contained content, like illustrations, diagrams, photos, and code listings, along with a caption. 
       Example: ::
       
         <figure>
           <img src="example.jpg" alt="Example">
           <figcaption>Figure caption.</figcaption>
         </figure>
       
   * - ``<div>``
     - Defines a division or section. 
       Example: ``<div>Content goes here...</div>``
   * - ``<span>``
     - Defines a section for inline elements. 
       Example: ``<span style="color: red;">Inline text</span>``
   * - ``<form>``
     - Defines an input form. 
       Example: ``<form action="/submit-form" method="post"> Form fields go here... </form>``
   * - ``<input>``
     - Defines an input field. 
       Example: ``<input type="text" name="fname">``
   * - ``<textarea>``
     - Defines a multiline input field. 
       Example: ``<textarea rows="4" cols="50"> Enter text here... </textarea>``
   * - ``<button>``
     - Defines a clickable button. 
       Example: ``<button onclick="alert('Button clicked!')">Click Me</button>``
   * - ``<select>``, ``<option>``
     - Defines a drop-down list, and the options within it. 
       Example: ::
       
         <select>
           <option value="volvo">Volvo</option>
           <option value="saab">Saab</option>
         </select>
       
   * - ``<label>``
     - Defines a label for an input element. 
       Example: ``<label for="username">Username:</label> <input type="text" id="username">``
   * - ``<fieldset>``, ``<legend>``
     - Groups related elements in a form and provides a caption for the group. 
       Example: ::
       
         <fieldset>
           <legend>Group Name</legend>
           <!-- Form elements go here... -->
         </fieldset>
       
   * - ``<table>``, ``<tr>``, ``<td>``
     - Defines a table, table rows, and table cells. 
       Example: ::
       
         <table>
           <tr>
             <td>Row 1, Cell 1</td>
             <td>Row 1, Cell 2</td>
           </tr>
           <tr>
             <td>Row 2, Cell 1</td>
             <td>Row 2, Cell 2</td>
           </tr>
         </table>
       
   * - ``<thead>``, ``<tbody>``, ``<tfoot>``
     - Defines the header, body, and footer sections of a table. 
       Example: ::
       
         <table>
           <thead>
             <tr>
               <th>Header 1</th>
               <th>Header 2</th>
             </tr>
           </thead>
           <tbody>
             <tr>
               <td>Data 1</td>
               <td>Data 2</td>
             </tr>
           </tbody>
           <tfoot>
             <tr>
               <td>Footer 1</td>
               <td>Footer 2</td>
             </tr>
           </tfoot>
         </table>
       
   * - ``<th>``
     - Defines a header cell in a table. 
       Example: ``<th>Header cell</th>``
   * - ``<caption>``
     - Defines a table caption. 
       Example: ``<caption>Table caption</caption>``
   * - ``<iframe>``
     - Embeds another HTML document within the current document. 
       Example: ``<iframe src="https://www.example.com" title="Example"></iframe>``
   * - ``<embed>``
     - Embeds external content or interactive content. 
       Example: ``<embed src="example.swf" type="application/x-shockwave-flash">``
   * - ``<object>``
     - Embeds multimedia content, including images, videos, and other content. 
       Example: ::
       
         <object data="example.mp4" width="320" height="240">
           <param name="autoplay" value="true">
         </object>
       
   * - ``<video>``
     - Embeds a video file. 
       Example: ::
       
         <video width="320" height="240" controls> 
           <source src="movie.mp4" type="video/mp4"> 
           Your browser does not support the video tag. 
         </video>
       
   * - ``<audio>``
     - Embeds an audio file. 
       Example: ::
       
         <audio controls> 
           <source src="audio.mp3" type="audio/mpeg"> 
           Your browser does not support the audio element. 
         </audio>
       
   * - ``<source>``
     - Specifies multiple media resources for media elements (``<video>`` and ``<audio>``). 
       Example: ::
       
         <video width="320" height="240" controls> 
           <source src="movie.mp4" type="video/mp4"> 
           <track src="subtitles_en.vtt" kind="subtitles" srclang="en" label="English">
           Your browser does not support the video tag. 
         </video>
       
   * - ``<track>``
     - Defines text tracks for media elements (e.g., subtitles). 
       Example: ``<track src="subtitles_en.vtt" kind="subtitles" srclang="en" label="English">``
   * - ``<canvas>``
     - Used for drawing graphics via scripting (usually JavaScript). 
       Example: ::
       
         <canvas id="myCanvas" width="200" height="100" style="border:1px solid #000000;">
           Your browser does not support the HTML5 canvas tag.
         </canvas>
       
   * - ``<svg>``
     - Defines vector-based graphics. 
       Example: ::
       
         <svg width="100" height="100">
           <circle cx="50" cy="50" r="40" stroke="black" stroke-width="3" fill="red" />
         </svg>
       
   * - ``<math>``
     - Defines a container for mathematical equations. 
       Example: ::
       
         <math>
           <msup>
             <mi>x</mi>
             <mn>2</mn>
           </msup>
         </math>
       
   * - ``<details>``, ``<summary>``
     - Defines additional details that the user can view or hide, with a visible heading. 
       Example: ::
       
         <details>
           <summary>Click to view details</summary>
           Additional details...
         </details>
       
   * - ``<dialog>``
     - Defines a dialog box or window. 
       Example: ``<dialog open> Dialog content... </dialog>``
   * - ``<template>``
     - Defines a template that can be cloned to produce similar elements. 
       Example: ::
       
         <template id="template">
           <p>Template content...</p>
         </template>
       
   * - ``<blockquote>``
     - Defines a section that is quoted from another source. 
       Example: ``<blockquote> Quoted text... </blockquote>``
   * - ``<cite>``
     - Defines the title of a work (e.g., a book, a song, a movie). 
       Example: ``<cite> Title of work </cite>``
   * - ``<q>``
     - Defines a short inline quotation. 
       Example: ``<p>The <q>quick</q> brown fox jumps over the lazy dog.</p>``
   * - ``<abbr>``
     - Defines an abbreviation or acronym. 
       Example: ``<abbr title="World Health Organization">WHO</abbr>``
   * - ``<code>``
     - Defines a piece of computer code. 
       Example: ``<code>console.log('Hello, world!');</code>``
   * - ``<kbd>``
     - Defines keyboard input. 
       Example: ``<kbd>Ctrl</kbd> + <kbd>C</kbd>``
   * - ``<samp>``
     - Defines sample output from a computer program. 
       Example: ``<samp>Output: 42</samp>``
   * - ``<var>``
     - Defines a variable in programming or in mathematics. 
       Example: ``<var>x</var> = <var>y</var> + 2``
   * - ``<mark>``
     - Defines marked or highlighted text. 
       Example: ``<p>Search term: <mark>example</mark></p>``
   * - ``<sup>``
     - Defines superscripted text. 
       Example: ``<p>E=mc<sup>2</sup></p>``
   * - ``<sub>``
     - Defines subscripted text. 
       Example: ``<p>H<sub>2</sub>O</p>``
   * - ``<time>``
     - Defines a specific time (or datetime). 
       Example: ``<time datetime="2024-06-30">June 30, 2024</time>``
   * - ``<data>``
     - Links the content with machine-readable data. 
       Example: ``<data value="42">The Answer to the Ultimate Question</data>``
   * - ``<meter>``
     - Represents a scalar measurement within a known range (e.g., disk usage). 
       Example: ::
       
         <meter value="75" min="0" max="100">75%</meter>
       
   * - ``<progress>``
     - Represents the progress of a task. 
       Example: ::
       
         <progress value="50" max="100">50%</progress>
       
   * - ``<ruby>``, ``<rt>``, ``<rp>``
     - Defines ruby annotations for East Asian typography. 
       Example: ::
       
         <ruby>
           漢 <rp>(</rp><rt>Kan</rt><rp>)</rp>
         </ruby>
       
   * - ``<bdo>``
     - Defines the direction of text display. 
       Example: ``<bdo dir="rtl">Right to Left</bdo>``
   * - ``<wbr>``
     - Defines a possible line-break. 
       Example: ``LongWordWithNoSpacesHere<wbr>ButBreaksHere``
   * - ``<figcaption>``
     - Defines a caption for a ``<figure>`` element. 
       Example: ::
       
         <figure>
           <img src="example.jpg" alt="Example">
           <figcaption>Figure caption.</figcaption>
         </figure>
       
   * - ``<slot>``
     - Defines a slot for inserting HTML content. 
       Example: ::
       
         <div>
           <p>This is a paragraph with a <slot>default value</slot>.</p>
         </div>
       
   * - ``<slot>`` (in Shadow DOM)
     - Defines a shadow tree insertion point. 
       Example: ::
       
         <div id="example">
           <slot name="example">Default text</slot>
         </div>
       
   * - ``<shadow>`` (deprecated)
     - Defines a shadow tree. 
       Example: ``<shadow></shadow>``
   * - ``<slot>`` (in Slotable DOM elements)
     - Defines a placeholder slot. 
       Example: ::
       
         <div>
           <slot name="example">Default value</slot>
         </div>
   * - ``<br>``
     - To insert a single line break within a text block. It does not have a closing tag. It is commonly used to separate lines of text within a paragraph or to start a new line in a poem or address.
       Example: ``<p>This is a line.<br>This is another line.</p>``

   * - ``<Non-Breaking Space (&nbsp;)>``
     - Represents a non-breaking space.
     - Example: `&nbsp;`
