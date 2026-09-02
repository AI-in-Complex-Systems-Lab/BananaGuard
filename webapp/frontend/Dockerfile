# Stage 1: Build the React application
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json first to leverage Docker cache
COPY package.json package-lock.json ./

# Install dependencies (using npm ci for reliable builds)
RUN npm ci

# Copy the rest of the application code
COPY . .

# Build the Vite application for production
RUN npm run build

# Stage 2: Serve the application with Nginx
FROM nginx:alpine

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy built static files from the builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config template (this allows Nginx to dynamically bind to the $PORT env var set by Cloud Run)
COPY default.conf.template /etc/nginx/templates/default.conf.template

# Cloud Run sets the PORT environment variable to 8080 by default. Expose 8080 for local testing.
ENV PORT=8080
EXPOSE 8080

# Nginx image automatically runs envsubst on templates in /etc/nginx/templates/ and outputs to /etc/nginx/conf.d/
CMD ["nginx", "-g", "daemon off;"]
