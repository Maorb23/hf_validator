import { defineRailway, project, service, volume, preserve } from "railway/iac";

export default defineRailway(() => {
  const data = volume("infiro-data", { region: "europe-west4-drams3a", sizeMB: 10240 });
  const web = service("infiro-web", {
    // Select this repository/branch in Railway; build context must be its root.
    build: { builder: "DOCKERFILE", dockerfilePath: "docker/web.Dockerfile" },
    start: "python deploy/web_start.py",
    replicas: { "europe-west4-drams3a": 1 },
    healthcheck: "/healthz",
    healthcheckTimeout: 300,
    deploy: { restartPolicyType: "ON_FAILURE", restartPolicyMaxRetries: 10,
              overlapSeconds: 0, drainingSeconds: 30 },
    volumeMounts: { "/data": data },
    env: {
      DJANGO_SECRET_KEY: preserve(),
      DJANGO_ALLOWED_HOSTS: preserve(),
      INFIRO_CREDENTIAL_KEYS: preserve(),
      INFIRO_ENV: "production",
      DJANGO_TRUST_PROXY_SSL_HEADER: "1",
      INFIRO_PRIVATE_ROOT: "/data/private",
      INFIRO_DATABASE: "/data/private/platform.sqlite3",
      INFIRO_LIVE_ENABLED: "0",
    },
  });
  return project("infiro", { resources: [web, data] });
});
