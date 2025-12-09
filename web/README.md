# 🎨 Document Scanner - Web Demo

## Design System

This web demo features a **minimalist, monochrome design** inspired by:

- **shadcn/ui** - Clean component design with subtle shadows
- **Apple Design** - Smooth animations and premium feel
- **Japanese Aesthetics** - Kanso (simplicity), Ma (negative space), Chōwa (harmony)

---

## ✨ Design Principles

### 1. **Kanso (簡素) - Simplicity**
- Pure black and white color palette
- Clean typography with Inter font
- Minimal visual noise
- Focus on content

### 2. **Ma (間) - Negative Space**
- Generous spacing between elements
- Breathing room for content
- Visual hierarchy through whitespace
- Uncluttered layout

### 3. **Chōwa (調和) - Harmony**
- Consistent design tokens
- Balanced proportions
- Unified visual language
- Cohesive experience

---

## 🎯 Features

### Navigation
- Fixed navigation bar with blur effect
- Smooth scroll to sections
- Minimal, functional links
- GitHub integration

### Hero Section
- Bold, attention-grabbing typography
- Animated badge with status indicator
- Code preview card with syntax highlighting
- Clear call-to-action buttons

### Stats Section
- Impactful metrics display
- Clean dividers
- Balanced layout

### Features Grid
- Hover animations
- Icon-based representation
- Card-based layout
- Responsive grid

### Interactive Demo
- Drag & drop image upload
- Example image buttons
- 6-step processing pipeline
- Interactive corner selection
- Adjustable thresholds
- Download functionality

### Algorithm Section
- Mathematical formulas
- Technical explanations
- Monospace code styling
- Parameter details

### Tech Stack
- Pill-based display
- Emoji icons
- Centered layout

### About Section
- Author information
- Quick links
- Grid layout

### Footer
- Minimal branding
- License links
- Clean typography

---

## 🎨 Color Palette

### Light Mode (Default)
```css
--background: #ffffff
--foreground: #09090b
--muted: #f4f4f5
--muted-foreground: #71717a
--border: #e4e4e7
```

### Dark Mode (Automatic)
```css
--background: #09090b
--foreground: #fafafa
--muted: #27272a
--muted-foreground: #a1a1aa
--border: #27272a
```

---

## 📐 Typography

- **Font Family**: Inter, SF Pro Display
- **Heading Weight**: 600-700
- **Body Weight**: 400-500
- **Letter Spacing**: -0.02em to -0.03em
- **Line Height**: 1.2 (headings), 1.6 (body)

---

## 🔄 Animations

### Fade Up Animation
```css
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
```

### Timing Function
```css
--ease-out: cubic-bezier(0.16, 1, 0.3, 1);
```

### Duration
- Fast: 150ms
- Base: 300ms
- Slow: 500ms

---

## 📱 Responsive Design

### Breakpoints
- Mobile: < 640px
- Tablet: 640px - 768px
- Desktop: > 768px

### Adaptations
- Single column layouts on mobile
- Adjusted typography sizes
- Full-width buttons
- Hidden navigation links on mobile

---

## 🚀 Quick Start

1. **Open the demo**:
   ```bash
   cd web
   open index.html
   ```

2. **Or use a local server**:
   ```bash
   python -m http.server 8000
   # Open http://localhost:8000
   ```

---

## 🛠️ Customization

### Change Colors
Edit `:root` variables in `styles.css`:
```css
:root {
    --background: #your-color;
    --foreground: #your-color;
    /* ... */
}
```

### Adjust Animations
Modify animation variables:
```css
--duration-fast: 150ms;
--duration-base: 300ms;
--duration-slow: 500ms;
```

### Change Typography
Update font-family:
```css
--font-sans: 'Your Font', sans-serif;
```

---

## 📊 Performance

- **No external dependencies** (except fonts)
- **Minimal CSS** (~700 lines)
- **Vanilla JavaScript** (~500 lines)
- **Fast loading** (< 100KB total)
- **Works offline** after initial load

---

## 🎓 Design Credits

This design is inspired by:
- [shadcn/ui](https://ui.shadcn.com/) - Component library
- [Apple Human Interface Guidelines](https://developer.apple.com/design/)
- [Japanese Design Principles](https://en.wikipedia.org/wiki/Japanese_aesthetics)
- [Vercel Design](https://vercel.com/design)
- [Stripe Checkout](https://stripe.com/)

---

## 📸 Screenshots

### Hero Section
![Hero](../docs/screenshots/hero.png)

### Features Section
![Features](../docs/screenshots/features.png)

### Demo Section
![Demo](../docs/screenshots/demo.png)

### Algorithm Section
![Algorithm](../docs/screenshots/algorithm.png)

---

## 🔮 Future Enhancements

- [ ] Dark mode toggle button
- [ ] More animation variations
- [ ] Custom cursor effects
- [ ] Page transition animations
- [ ] Particle background effects
- [ ] 3D visualization of Hough space

---

<div align="center">

**Designed with 簡素 (simplicity) • 間 (space) • 調和 (harmony)**

Made with ❤️ for CV583

</div>
