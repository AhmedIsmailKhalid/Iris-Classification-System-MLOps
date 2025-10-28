# System Design Decisions

## **Decision 1: Monorepo vs Multi-repo**
**Choice:** Monorepo with `backend/` and `frontend/` directories
**Reasoning:**
- Easier to manage for single project
- Shared documentation
- Simpler CI/CD coordination
- Better for portfolio presentation

## **Decision 2: Poetry vs pip**
**Choice:** Poetry 
**Reasoning:**
- Better dependency resolution
- Lock file for reproducibility
- Professional standard
- Easier virtual environment management
- pyproject.toml is modern standard

## **Decision 3: React vs Vue**
**Choice:** React 
**Reasoning:**
- Larger job market
- More FAANG companies use React
- Better ecosystem
- Stronger resume signal

## **Decision 4: Tailwind vs Material-UI**
**Choice:** Tailwind CSS
**Reasoning:**
- More control over design
- Lighter bundle size
- Industry momentum
- Shows CSS knowledge

## **Decision 5: State Management**
**Choice:** React useState (no Redux/Zustand for now)
**Reasoning:**
- Simple application, no need for complex state
- Reduces dependencies
- Easier to understand for reviewers
- Can add later if needed

## **Decision 6: API Versioning**
**Choice:** URL versioning (`/api/v1/predict`)
**Reasoning:**
- Clear and explicit
- Easy to maintain multiple versions
- Industry standard
- Preparation for future scaling

## **Decision 7: CORS Strategy**
**Choice:** Whitelist specific origins
**Reasoning:**
- Security best practice
- Production-ready approach
- Easy to configure per environment