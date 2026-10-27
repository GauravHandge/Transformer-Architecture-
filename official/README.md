# Transformer Explainer: Frontend & Development Pipeline

This directory contains the SvelteKit and Vite frontend source code for the **Transformer Architecture & Explainer** project maintained by **Gaurav Handge**.

The production build of this frontend is served directly by the Python FastAPI backend on port `8080`.

---

## 🛠️ Development & Building

### Prerequisites

* Node.js v20+
* NPM v10+

### Development Server

To run the standalone frontend development server with hot-reloading:

```bash
npm install
npm run dev
```

The dev server will start at `http://127.0.0.1:5173`.

### Production Build

To compile the frontend into the static bundle (`build/`):

```bash
npm run build
```

The compiled assets can be served by the FastAPI Python backend in the root `frontend/` directory.

---

## 👤 Author & Maintainer

* **Gaurav Handge**
  * GitHub: [@GauravHandge](https://github.com/GauravHandge)
  * Repository: [Transformer-Architecture-](https://github.com/GauravHandge/Transformer-Architecture-)
  * Email: [gaurav.handge@voyatekgroup.com](mailto:gaurav.handge@voyatekgroup.com)

---

## 📄 License

This project is licensed under the [MIT License](../LICENSE).
