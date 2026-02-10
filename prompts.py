# Prompts (defined as in previous responses)
START_PROMPT = """Your task is to create a website consisting of three separate files: index.html, styles.css, and script.js, based on the given specifications.

The index.html file should contain the basic structure and content of the website. Think about fundamental HTML elements to organize content into sections. For example, a typical structure might include:

```html
<!DOCTYPE html>
<html>
<head>
    <title>[Website Title - AI's Choice]</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <!-- Website Header Content (e.g., navigation, logo) -->
    </header>
    <main>
        <!-- Main Website Content -->
    </main>
    <footer>
        <!-- Website Footer Content (e.g., copyright, links) -->
    </footer>
    <script src="script.js"></script>
</body>
</html>
```

The styles.css file should handle all the styling to make the website visually appealing, responsive across different screen sizes, and user-friendly. Consider using CSS for layout, typography, colors, and responsiveness using media queries. Example CSS concepts to think about:

```css
/* styles.css - Example CSS Concepts */
body {
    font-family: sans-serif; /* Example: Choose a readable font */
    margin: 0; /* Example: Reset default margins */
}

/* Example: Style the header, main content, and footer */
header { /* ... styles for header ... */ }
main { /* ... styles for main content ... */ }
footer { /* ... styles for footer ... */ }

/* Example: Media query for responsiveness (adjust styles for smaller screens) */
@media (max-width: 768px) {
    /* Styles to apply when screen width is 768px or less */
}
```

The script.js file should implement interactive features to enhance the user experience. Think about adding simple interactions like:

```javascript
// script.js - Example Interactive Concepts
document.addEventListener('DOMContentLoaded', () => {
    // Example: Add interactivity after the page has loaded
    // You could add event listeners to buttons, create dynamic content, etc.
});
```

Use your creative freedom to choose a website topic and design. The goal is to create a functional, visually appealing, and responsive website demonstrating good HTML, CSS, and JavaScript practices. It could be a website about anything you can imagine – a blog, a portfolio, a simple tool, a landing page, etc.

Please ensure that the HTML, CSS, and JavaScript code in their respective files are well-structured, efficiently organized, and properly commented for readability and maintainability.

The output should be provided in three separate fenced code blocks, clearly labeled with their respective file extensions: "HTML", "CSS", and "JavaScript". For example:

```html
<!-- HTML content here -->
```
```css
/* styles.css content here */
```
```javascript
// script.js content here */
```
"""

IMPROVE_PROMPT = """As a professional web developer, implement these specific improvements to the existing website:

REQUIRED IMPROVEMENTS:
[USER INSTRUCTIONS WILL BE INSERTED HERE]

Current website code:

HTML:
```
[CURRENT_HTML]
```

CSS:
```
[CURRENT_CSS]
```

JavaScript:
```
[CURRENT_JS]
```

Make substantial, meaningful changes to implement each requested improvement. Focus on creating a significantly better version of the website.

Your response must include complete, updated code for all three files:
1. Full HTML code (index.html)
2. Full CSS code (styles.css)
3. Full JavaScript code (script.js)

Format each file in its own code block with the appropriate language tag.
"""

ANALYZE_CONTENT_PROMPT = """You are now acting as a Website Content Consultant, focusing exclusively on the content of the website.

Your task is to analyze the current website content and suggest specific improvements to make the content more engaging, comprehensive, and valuable for users. In each iteration, the goal is to add more meaningful content to the website.

Review the HTML structure and content:
- Is the content comprehensive and detailed enough for the website's purpose?
- Are there sections that should be expanded with more information?
- Could additional content elements improve the user experience (e.g., FAQs, examples, details)?
- Are there opportunities to add more engaging content (stories, examples, visuals, etc.)?
- Is the content well-organized and easy to follow?
- Does the content meet the needs of the target audience?

For each suggested improvement:
1. Describe what specific content should be added or expanded
2. Explain why this additional content would benefit users
3. Provide a brief outline of the content to add (be specific and detailed)

Focus ONLY on content recommendations - do not suggest visual styling or JavaScript functionality changes.

Here is the current website HTML:

```html
[CURRENT_HTML]
```

Please provide 3-5 concrete, specific content improvements that would meaningfully enhance this website in this iteration.
"""

SUGGESTION_SELECTOR_PROMPT = """You are now acting as a Website Improvement Suggestion Selector. Your task is to review a list of website improvement suggestions generated by a Website Analyzer and select the most valuable and impactful suggestions to implement in the next iteration of website development.

You will be given:

A list of Website Improvement Suggestions: This list is structured and categorized by area (HTML, CSS, JavaScript, Content). Each suggestion includes:

- **Suggestion Title:** A brief description of the suggested improvement.
- **Benefit:** An explanation of why implementing this suggestion is beneficial.
- **Implementation Notes:** Brief notes on how the suggestion could be implemented.

[Optional: Iteration Number]: You may be told the current iteration number in the website development process. This can help you prioritize suggestions (e.g., focus on foundational HTML in early iterations, and styling/interactivity in later iterations).

Your Goal:

Select a limited number of the most impactful suggestions from the list to be implemented in the next iteration. Aim to choose suggestions that will provide the greatest overall improvement to the website in terms of:

- **User Experience:** Will this improvement make the website more user-friendly, accessible, and enjoyable to use?
- **Functionality:** Will this improve the website's features and capabilities?
- **Visual Appeal and Design:** Will this enhance the website's aesthetics and visual presentation?
- **Technical Quality:** Will this improve the website's code structure, performance, maintainability, or SEO?
- **Efficiency and Feasibility:** Are the suggested improvements relatively easy and efficient to implement in the current iteration? Prioritize suggestions that are achievable in a reasonable timeframe.

Consider these factors when selecting suggestions:

- **Impact vs. Effort:** Prioritize suggestions that offer a high impact for a reasonable amount of implementation effort.
- **Dependencies:** Are there any dependencies between suggestions? Should some suggestions be implemented before others?
- **Balance:** Aim for a balance of improvements across different areas (HTML, CSS, JavaScript, Content) if possible, unless there is a clear priority area in a given iteration.
- **Iteration Stage (if provided):** In early iterations, prioritize foundational improvements (HTML structure, basic CSS). In later iterations, focus on enhancements, polish, and advanced features.

Output:

Provide a list of just the titles of the suggestions you have selected for implementation. List them in order of priority if possible (most important first). You can also briefly explain why you selected each suggestion if you feel it's helpful to justify your choices.

Example Input (List of Suggestions - like output from the Analyzer Prompt):

HTML (index.html) Suggestions:
- **Suggestion 1:** Use Semantic Tags: The <div id="header"> could be replaced with the <header> semantic tag for better structure and accessibility.
    - **Benefit:** Improves semantic meaning and accessibility for screen readers and SEO.
    - **Implementation:** Replace <div id="header"> with <header> and </div> with </header>.
- **Suggestion 2:** Add Alt Text to Images: Some images are missing alt attributes.
    - **Benefit:** Improves accessibility for visually impaired users and SEO.
    - **Implementation:** Add descriptive alt attributes to all <img> tags.

CSS (styles.css) Suggestions:
- **Suggestion 1:** Improve Mobile Responsiveness: The website doesn't seem to adapt well to mobile screens.
    - **Benefit:** Ensures a good user experience on mobile devices, crucial for modern websites.
    - **Implementation:** Add media queries in styles.css to adjust layout, font sizes, and element sizes for smaller screen widths.
- **Suggestion 2:** Use a Consistent Color Palette: The color scheme is a bit inconsistent.
    - **Benefit:** Improves visual coherence and professionalism.
    - **Implementation:** Define a limited color palette and apply it consistently across the website.

JavaScript (script.js) Suggestions:
- **Suggestion 1:** Add Form Validation (if a form exists): If there is a form, add client-side validation.
    - **Benefit:** Improves user experience and reduces server-side load.
    - **Implementation:** Use JavaScript to validate form inputs before submission.

Example Output (Selected Suggestion Titles):

- Use Semantic Tags
- Improve Mobile Responsiveness
- Add Alt Text to Images

**[Optional - Example Output with Justification]:**

- **Use Semantic Tags:** Selected because it's a foundational HTML improvement with good impact on accessibility and SEO, and relatively easy to implement.
- **Improve Mobile Responsiveness:** Crucial for modern websites to be mobile-friendly for good user experience. High impact.
- **Add Alt Text to Images:** Important for accessibility and SEO, and straightforward to implement.

Please provide your selected suggestion titles in the requested format."""

ANALYZE_VISUAL_PROMPT = """You are now acting as a Website Visual Design Consultant, focusing exclusively on the visual appearance and styling of the website.

Your task is to analyze the current website's CSS and visual elements and suggest specific improvements to enhance its visual appeal, user experience, and responsiveness.

Review the HTML structure and CSS styling:
- Is the visual design aesthetically pleasing and appropriate for the website's purpose?
- Is the website responsive and does it display well on different screen sizes?
- Is the typography (font choices, sizes, line heights) effective and readable?
- Is the color scheme appropriate, accessible, and visually appealing?
- Could the layout be improved for better visual hierarchy and organization?
- Are there opportunities to add visual elements that would enhance the user experience?
- Could animations or transitions improve the interface?

For each suggested improvement:
1. Describe the specific visual/CSS change
2. Explain why this improvement would enhance the user experience
3. Provide a brief implementation approach (specific CSS properties to modify)

Focus ONLY on visual and CSS recommendations - do not suggest content additions or JavaScript functionality changes.

Here is the current website HTML and CSS:

```html
[CURRENT_HTML]
```

```css
[CURRENT_CSS]
```

Please provide 3-5 concrete, specific visual design improvements that would meaningfully enhance this website in this iteration.
"""

ANALYZE_JS_PROMPT = """You are now acting as a Website Interactivity Consultant, focusing exclusively on JavaScript functionality and interactivity.

Your task is to analyze the current website's JavaScript code and suggest specific improvements to enhance interactivity, user engagement, and functionality.

Review the HTML structure and JavaScript code:
- Could additional interactive features improve user engagement?
- Are there opportunities to add functionality that would help users (e.g., filtering, sorting, calculations)?
- Could interactive elements make the content more engaging or easier to navigate?
- Are there any UI/UX improvements that require JavaScript implementation?
- Could animations or transitions improve the user experience?
- Are there ways to make the site more dynamic and responsive to user actions?

For each suggested improvement:
1. Describe the specific JavaScript functionality to add
2. Explain why this feature would benefit users
3. Provide a brief implementation approach (what elements to target, what events to listen for)

Focus ONLY on JavaScript and interactivity recommendations - do not suggest content additions or pure CSS styling changes.

Here is the current website HTML and JavaScript:

```html
[CURRENT_HTML]
```

```javascript
[CURRENT_JS]
```

Please provide 3-5 concrete, specific JavaScript functionality improvements that would meaningfully enhance this website in this iteration.
"""