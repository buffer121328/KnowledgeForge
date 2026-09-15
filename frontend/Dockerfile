FROM node:22-alpine AS builder

ARG NPM_REGISTRY=https://registry.npmjs.org

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm,sharing=locked \
    node -e "const fs=require('fs');const lock=JSON.parse(fs.readFileSync('package-lock.json','utf8'));const scrub=(value)=>{if(!value||typeof value!=='object')return;if(typeof value.resolved==='string'&&value.resolved.includes('artifactory.hundsun.com'))delete value.resolved;for(const child of Object.values(value))scrub(child)};scrub(lock);fs.writeFileSync('package-lock.json',JSON.stringify(lock))" \
    && npm ci --cache=/root/.npm --registry="$NPM_REGISTRY"

COPY frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine AS runtime

COPY frontend/nginx.conf /etc/nginx/nginx.conf
COPY --from=builder --chown=nginx:nginx /app/dist /usr/share/nginx/html

USER nginx

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["wget", "-qO-", "http://127.0.0.1:8080/healthz"]

CMD ["nginx", "-g", "daemon off;"]
